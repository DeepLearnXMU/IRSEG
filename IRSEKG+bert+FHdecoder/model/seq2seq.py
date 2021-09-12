import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
from random import shuffle
import torch.nn.functional as F
from model.generator import Beam
from datatool.data import DocField, DocDataset, DocIter
from model import modeling_bert


# GRU单元的类
class GRUCell(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)  # 记忆消除门的初始化：输入维度+隐向量的维度，输出隐向量维度
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)  # 隐状态权重门的初始化：输入维度+隐向量的维度，输出隐向量维度

        self.c = nn.Linear(x_dim, h_dim, True)  # 把input变成隐状态
        self.u = nn.Linear(h_dim, h_dim, True)  # 当前隐状态

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(x) + r * self.u(h))

        new_h = z * h + (1 - z) * u
        return new_h


class SGRU(nn.Module):
    def __init__(self, s_emb, e_emb, sh_dim, eh_dim, label_dim):
        super(SGRU, self).__init__()

        g_dim = sh_dim
        self.s_gru = GRUCell(s_emb + sh_dim + label_dim + eh_dim + g_dim, sh_dim)  # 句子编码、上图层句消息、标签、上图层实体消息、上图层全局隐向量

        self.e_gru = GRUCell(e_emb + eh_dim + sh_dim + label_dim + g_dim, eh_dim)  # 实体编码、上图层句消息、标签、上图层全局隐向量

        self.g_gru = GRUCell(sh_dim + eh_dim, g_dim)  # 上图层句消息、上图层实体消息

    # s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))
    def forward(self, it, h, g, mask):
        '''
        :param it: B T 2H
        :param h: B T H
        :param g: B H
        :return:
        '''

        si, ei = it
        sh, eh = h
        smask, wmask = mask

        # update sentence node
        g_expand_s = g.unsqueeze(1).expand_as(sh)
        x = torch.cat((si, g_expand_s), -1)
        new_sh = self.s_gru(x, sh)

        # update entity node
        g_expand_e = g.unsqueeze(1).expand(eh.size(0), eh.size(1), g.size(-1))
        x = torch.cat((ei, g_expand_e), -1)
        new_eh = self.e_gru(x, eh)

        new_sh.masked_fill_((smask == 0).unsqueeze(2), 0)
        new_eh.masked_fill_((wmask == 0).unsqueeze(2), 0)

        # update global
        sh_mean = new_sh.sum(1) / smask.float().sum(1, True)
        eh_mean = new_eh.sum(1) / (wmask.float().sum(1, True) + 1)

        mean = torch.cat((sh_mean, eh_mean), -1)
        new_g = self.g_gru(mean, g)

        return new_sh, new_eh, new_g


class GRNGOB(nn.Module):
    def __init__(self, s_emb, e_emb, s_hidden, e_hidden, label_dim, dp=0.1, layer=2, agg='sum'):
        super(GRNGOB, self).__init__()
        self.layer = layer
        self.dp = dp

        self.slstm = SGRU(s_emb, e_emb, s_hidden, e_hidden, label_dim)

        self.s_hid = s_hidden
        self.e_hid = e_hidden

        self.agg = agg

        # sent_neighbour_sent
        self.s_orderemb = nn.Embedding(2, label_dim)
        self.s_edgeemb = nn.Embedding(11, label_dim)

        self.gate0 = nn.Linear(s_hidden + s_hidden + label_dim + label_dim, s_hidden)

        # sent_neighbour_ent
        self.edgeemb = nn.Embedding(4, label_dim)
        self.gate1 = nn.Linear(s_hidden + e_hidden + label_dim, e_hidden + label_dim)
        self.gate2 = nn.Linear(s_hidden + e_hidden + label_dim, s_hidden + label_dim)
        self.gate3 = nn.Linear(2*e_hidden, e_hidden)    #新加的门单元：实体和实体之间
        #self.S = nn.Parameter(torch.zeros(outdim))     #公式中用于调整先验的参数Sa(向量版)
        self.S = nn.Parameter(torch.tensor([0.]))      #标量版

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    # para, hn = self.encoder(sentences, sents_mask, entity_emb, words_mask, elocs)
    def forward(self, sent, smask, word, wmask, elocs, slocs, ecom):
        '''
        :param wmask: Batch* Entities
        :param smask: Batch* Sentences
        :param word: Batch* Entities H
        :param sent: Batch* Sentences H
        :return:
        '''
        batch = sent.size(0)
        snum = smask.size(1)
        wnum = wmask.size(1)

        # batch sent_num word_num
        # B S E
        matrix = sent.new_zeros(batch, snum, wnum).long()  # Batch × Sentences × Entity 实体在句子里的角色

        for ib, eloc in enumerate(elocs):
            for ixw, loc in enumerate(eloc):
                for aftersf_ixs, r in loc:
                    matrix[ib, aftersf_ixs, ixw] = r

        mask_se = (matrix != 0).float()  # Batch × Sentences × Entity 句子、实体邻接矩阵
        mask_se_t = mask_se.transpose(1, 2)

        # B S E H
        label_emb = self.edgeemb(matrix)
        label_emb_t = label_emb.transpose(1, 2)

        # B S S
        # connect two sentence if thay have at least one same entity

        # *********NEW s2smatrix ***********
        s2smatrix = torch.zeros(batch, snum, snum).cuda().byte()
        s2sorder = torch.zeros(batch, snum, snum).cuda().byte()
        for ib, sloc in enumerate(slocs):
            for triple in sloc:
                s2smatrix[ib, triple[0], triple[1]] = int(triple[2] * 10)
                s2smatrix[ib, triple[1], triple[0]] = int(10 * (1.0001 - triple[2]))
                if (triple[2] > 0.5):
                    s2sorder[ib, triple[0], triple[1]] = 1
                    s2sorder[ib, triple[1], triple[0]] = 0
                else:
                    s2sorder[ib, triple[0], triple[1]] = 0
                    s2sorder[ib, triple[1], triple[0]] = 1

        eye = torch.eye(snum).byte().cuda()
        # eye = torch.eye(snum).byte()
        # print("eye:",eye)

        s2smask = torch.matmul(mask_se, mask_se_t)
        s2smask = s2smask != 0

        s2s = smask.new_ones(snum, snum)
        eyemask = (s2s - eye).unsqueeze(0)

        s2smask = s2smask * eyemask
        s2smask = s2smask & smask.unsqueeze(1)
        s2smask = s2smask.float()

        s2sorder = s2sorder * eyemask
        s2sorder = s2sorder & smask.unsqueeze(1)
        s2sorder = s2sorder.long()
        order_emb = self.s_orderemb(s2sorder)
        order_emb = order_emb * s2smask.unsqueeze(-1)

        s2smatrix = s2smatrix * eyemask
        s2smatrix = s2smatrix.long()
        value_emb = self.s_edgeemb(s2smatrix)
        value_emb = value_emb * s2smask.unsqueeze(-1)

        # Batch× Entity× Entity
        e2ematrix = torch.zeros(batch, wnum, wnum).cuda().float()  # 实体对的基于wordnet/wiki的相似度矩阵
        e2emask = torch.zeros(batch, wnum, wnum).cuda().float()    # 记录哪些实体对是有关系的（1有0无）
        for b_index in range(batch):
            ent_rel_of_sent = ecom[b_index]
            for index in range(len(ent_rel_of_sent)):
                relation = ent_rel_of_sent[index]
                if relation[2] < 0.5:
                   relation[2] = 0
                e2ematrix[b_index][relation[0]][relation[1]] = relation[2]
                e2ematrix[b_index][relation[1]][relation[0]] = relation[2]
                e2emask[b_index][relation[0]][relation[1]] = [0., 1.][relation[2] > 0]
                e2emask[b_index][relation[1]][relation[0]] = e2emask[b_index][relation[0]][relation[1]]
        e2ematrix = e2ematrix / (e2ematrix.sum(-1) + 1e-9).unsqueeze(-1)    # 归一化
        e2ematrix.masked_fill_((e2ematrix == 0), 1e-9)                      # 替换0防止Sa的学习出错

        s_h = torch.zeros_like(sent)
        g_h = sent.new_zeros(batch, self.s_hid)
        e_h = sent.new_zeros(batch, wnum, self.e_hid)


        for i in range(self.layer):
            # 1.aggregation
            # s_neigh_s_h = self.mean(s_h, s2smatrix)
            # s_neigh_s_h = self.sum(s_h, s2smatrix)

            # B S E H
            if self.agg == 'gate':

                # 句对句间
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, snum, self.s_hid)
                s_h_expand_t = s_h.unsqueeze(1).expand(batch, snum, snum, self.s_hid)

                s_h_expand_order = torch.cat((s_h_expand_t, order_emb), -1)
                s_h_expand_order_value = torch.cat((s_h_expand_order, value_emb), -1)
                s_s_l = torch.cat((s_h_expand, s_h_expand_order_value), -1)

                gs = torch.sigmoid(self.gate0(s_s_l))
                # s_neigh_s_h = s_h_expand_order_value * gs * s2smask.unsqueeze(3)
                s_neigh_s_h = s_h_expand_t * gs * s2smask.unsqueeze(3)
                s_neigh_s_h = s_neigh_s_h.sum(2)

                # 句对词
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, wnum, self.s_hid)
                e_h_expand = e_h.unsqueeze(1).expand(batch, snum, wnum, self.e_hid)

                # 带上了label信息的实体的隐状态
                e_h_expand_edge = torch.cat((e_h_expand, label_emb), -1)

                s_e_l = torch.cat((s_h_expand, e_h_expand_edge), -1)
                g = torch.sigmoid(self.gate1(s_e_l))

                s_neigh_e_h = e_h_expand_edge * g * mask_se.unsqueeze(3)
                s_neigh_e_h = s_neigh_e_h.sum(2)

                # 词对句
                s_h_expand = s_h.unsqueeze(1).expand(batch, wnum, snum, self.s_hid)
                s_h_expand_edge = torch.cat((s_h_expand, label_emb_t), -1)

                e_h_expand = e_h.unsqueeze(2).expand(batch, wnum, snum, self.e_hid)

                e_s_l = torch.cat((e_h_expand, s_h_expand_edge), -1)
                g2 = torch.sigmoid(self.gate2(e_s_l))

                e_neigh_s_h = s_h_expand_edge * g2 * mask_se_t.unsqueeze(3)
                e_neigh_s_h = e_neigh_s_h.sum(2)

                # 实体间的gate
                e_h_expand = e_h.unsqueeze(-2).expand(batch, wnum, wnum, self.e_hid)		#[[[A],[A],[A]],[[B],[B],[B]],[[C],[C],[C]]]
                e_neigh_h = e_h_expand.transpose(-2,-3)					#[[[A],[B],[C]],[[A],[B],[C]],[[A],[B],[C]]]
                e_neigh_e_h = torch.cat((e_h_expand,e_neigh_h),dim=-1)			#[[[AA],[AB],[AC]],[[BA],[BB],[BC]],[[CA][CB][CC]]]
                es = self.gate3(e_neigh_e_h)       #将上述扩展维度后的Batch×Entity×Entity经转化成gate的输入
                g3 = torch.sigmoid(es + (self.S * torch.log(e2ematrix).unsqueeze(-1)))  #即lattice论文的公式
                #g3 = torch.sigmoid(es)  #即lattice论文的公式
                #two layer

                e_neigh_e_h = e_h_expand * e2emask.unsqueeze(-1) * g3                    #[[[A],[B],[C]],[[A],[B],[C]],[[A],[B],[C]]]与mask和gate元素积
                e_neigh_e_h = e_neigh_e_h.sum(2)                                        # 维度2（第3级维度）上进行加和，变回Batch×Entity×隐向量维度

            s_input = torch.cat((sent, s_neigh_s_h, s_neigh_e_h), -1)
            e_input = torch.cat((word, e_neigh_e_h, e_neigh_s_h), -1)

            # 2.update
            s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))

        if self.dp > 0:
            s_h = F.dropout(s_h, self.dp, self.training)

        return s_h, g_h

    def teacherForward(self, sent, smask, word, wmask, elocs, slocs):
        batch = sent.size(0)
        snum = smask.size(1)
        wnum = wmask.size(1)

        # batch sent_num word_num
        # B S E
        matrix = sent.new_zeros(batch, snum, wnum).long()  # Batch × Sentences × Entity 实体在句子里的角色

        for ib, eloc in enumerate(elocs):
            for ixw, loc in enumerate(eloc):
                for aftersf_ixs, r in loc:
                    matrix[ib, aftersf_ixs, ixw] = r

        mask_se = (matrix != 0).float()  # Batch × Sentences × Entity 句子、实体邻接矩阵
        mask_se_t = mask_se.transpose(1, 2)

        # B S E H
        label_emb = self.edgeemb(matrix)
        label_emb_t = label_emb.transpose(1, 2)

        # B S S
        # connect two sentence if thay have at least one same entity

        # *********NEW s2smatrix ***********
        s2smatrix = torch.zeros(batch, snum, snum).cuda().float()
        for ib, sloc in enumerate(slocs):
            for triple in sloc:
                s2smatrix[ib, triple[0], triple[1]] = triple[2]
                s2smatrix[ib, triple[1], triple[0]] = 1.000 - triple[2]

        s_h = torch.zeros_like(sent)
        g_h = sent.new_zeros(batch, self.s_hid)
        e_h = sent.new_zeros(batch, wnum, self.e_hid)

        for i in range(self.layer):
            # 1.aggregation
            s_neigh_s_h = self.sum(s_h, s2smatrix)

            # B S E H
            if self.agg == 'gate':
                # 句对词
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, wnum, self.s_hid)
                e_h_expand = e_h.unsqueeze(1).expand(batch, snum, wnum, self.e_hid)

                # 带上了label信息的实体的隐状态
                e_h_expand_edge = torch.cat((e_h_expand, label_emb), -1)

                s_e_l = torch.cat((s_h_expand, e_h_expand_edge), -1)

                s_neigh_e_h = e_h_expand_edge * g * mask_se.unsqueeze(3)
                s_neigh_e_h = s_neigh_e_h.sum(2)

                # 词对句
                s_h_expand = s_h.unsqueeze(1).expand(batch, wnum, snum, self.s_hid)
                s_h_expand_edge = torch.cat((s_h_expand, label_emb_t), -1)

                e_h_expand = e_h.unsqueeze(2).expand(batch, wnum, snum, self.e_hid)

                e_s_l = torch.cat((e_h_expand, s_h_expand_edge), -1)
                g2 = torch.sigmoid(self.gate2(e_s_l))

                e_neigh_s_h = s_h_expand_edge * g2 * mask_se_t.unsqueeze(3)
                e_neigh_s_h = e_neigh_s_h.sum(2)

            s_input = torch.cat((sent, s_neigh_s_h, s_neigh_e_h), -1)
            e_input = torch.cat((word, e_neigh_s_h), -1)

            # 2.update
            s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))

        if self.dp > 0:
            s_h = F.dropout(s_h, self.dp, self.training)

        return s_h, g_h


class OrderPredictor(nn.Module):
    def __init__(self, dim_mlp, sent_hidden, dropout=0.1):
        super(OrderPredictor, self).__init__()
        self.dp = dropout
        self.d_mlp = dim_mlp
        self.s_hid = sent_hidden

        self.MLP = nn.Sequential(nn.Linear(self.s_hid * 2, self.d_mlp, False), nn.ReLU(), nn.Dropout(self.dp),
                                 nn.Linear(self.d_mlp, self.d_mlp, False), nn.ReLU(), nn.Dropout(self.dp))
        self.PredictWeight = nn.Linear(self.d_mlp, 1, False)  # 正向是1，反向是0

    def forward(self, sentences):
        order_score = self.PredictWeight(self.MLP(sentences))
        order_prob = torch.sigmoid(order_score)
        return order_prob


class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio
        self.d_emb = args.d_emb
        self.sen_enc_type = args.senenc

        self.src_embed = nn.Embedding(args.doc_vocab,100)  # 把document的词汇编码成embedding
        self.bert = modeling_bert.BertModel.from_pretrained('bert-base-uncased')

        self.sen_enc = nn.LSTM(self.d_emb, args.d_rnn // 2, bidirectional=True,
                               batch_first=True)  # 把document的词汇的embedding编码为句子级别的向量

        self.entityemb = args.entityemb

        self.encoder = GRNGOB(s_emb=args.d_rnn,
                              e_emb=args.d_emb if self.entityemb == 'glove' else args.d_rnn,
                              s_hidden=args.d_rnn,
                              e_hidden=args.ehid, label_dim=args.labeldim,
                              layer=args.gnnl, dp=args.gnndp, agg=args.agg)

        d_mlp = args.d_mlp
        self.linears = nn.ModuleList([nn.Linear(args.d_rnn, d_mlp),
                                      nn.Linear(args.d_rnn*2  , d_mlp),
                                      nn.Linear(d_mlp, 1)])
        self.decoder = nn.LSTM(args.d_rnn, args.d_rnn, batch_first=True)
        self.critic = None

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

        # future ffn
        self.future = nn.Sequential(nn.Linear(args.d_rnn * 2, args.d_rnn * 2, False), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(args.d_rnn * 2, args.d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        self.w3 = nn.Linear(args.d_pair, 2, False)
        self.hist_left1 = nn.Sequential(nn.Linear(args.d_rnn * 2, args.d_rnn * 2, False), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(args.d_rnn * 2, args.d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        # for sind, l2 half dim
        self.hist_left2 = nn.Sequential(nn.Linear(args.d_rnn * 2, args.d_rnn * 2, False), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(args.d_rnn * 2, args.d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        self.wleft1 = nn.Linear(args.d_pair, 2, False)
        self.wleft2 = nn.Linear(args.d_pair, 2, False)
        d_label = 2
        # new key
        d_pair_posi = args.d_pair + d_label
        self.pw_k = nn.Linear(d_pair_posi * 4, args.d_rnn, False)
        self.newkey = nn.Linear(args.d_rnn * 2, args.d_rnn, False)
        self.pw_e = nn.Linear(args.d_rnn, 1, False)

    def equip(self, critic):
        self.critic = critic

    def encode_history(self, paragraph, g):
        batch, num, hdim = paragraph.size()

        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)

        input = torch.cat((para_unq2, para_unq1), -1)
        rela_left1 = self.hist_left1(input)
        rela_left2 = self.hist_left2(input)
        return rela_left1, rela_left2

    def rela_encode(self, paragraph, g):
        batch, num, hdim = paragraph.size()
        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)
        # B N N H
        input = torch.cat((para_unq2, para_unq1), -1)
        return self.future(input)

    def rela_pred(self, paragraph, g):
        rela_vec = self.rela_encode(paragraph, g)
        rela_p = F.softmax(self.w3(rela_vec), -1)
        rela_vec_diret = torch.cat((rela_vec, rela_p), -1)

        hist_left1, hist_left2 = self.encode_history(paragraph, g)
        left1_p = F.softmax(self.wleft1(hist_left1), -1)
        left2_p = F.softmax(self.wleft2(hist_left2), -1)

        hist_vec_left1 = torch.cat((hist_left1, left1_p), -1)
        hist_vec_left2 = torch.cat((hist_left2, left2_p), -1)

        # prob, label = torch.topk(rela_p, 1)
        return (left1_p, left2_p, rela_p), rela_vec_diret, hist_vec_left1, hist_vec_left2

    def key(self, paragraph, rela_vec):
        rela_mask = rela_vec.new_ones(rela_vec.size(0), rela_vec.size(1), rela_vec.size(2)) \
                    - torch.eye(rela_vec.size(1)).cuda().unsqueeze(0)

        rela_vec_mean = torch.sum(rela_vec * rela_mask.unsqueeze(3), 2) / rela_mask.sum(2, True)
        pre_key = torch.cat((paragraph, rela_vec_mean), -1)
        key = self.linears[1](pre_key)
        return key

    def forward(self, src_and_len, tgt_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, ecom, blocs):

        document_matrix, GRN_sents, hcn, GRNkey = self.encode(src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, ecom, blocs)
        target, tgt_len = tgt_and_len
        batch, num = target.size()

        tgt_len_less = tgt_len
        target_less = target

        target_mask = torch.zeros_like(target_less).byte()
        pointed_mask_by_target = torch.zeros_like(target).byte()

        # relative order loss
        rela_vec = self.rela_encode(document_matrix, hcn[0])
        score = self.w3(rela_vec)

        # B N N 2
        logp_rela = F.log_softmax(score, -1)

        truth = torch.tril(logp_rela.new_ones(num, num)).long().unsqueeze(0).expand(batch, num, num)

        logp_rela = logp_rela[torch.arange(batch).unsqueeze(1), target]
        logp_rela = logp_rela[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                              torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        loss_rela = self.critic(logp_rela.view(-1, 2), truth.contiguous().view(-1))

        # history loss
        rela_hist_left1, rela_hist_left2 = self.encode_history(document_matrix, hcn[0])
        score_left1 = self.wleft1(rela_hist_left1)
        score_left2 = self.wleft2(rela_hist_left2)

        logp_left1 = F.log_softmax(score_left1, -1)
        logp_left2 = F.log_softmax(score_left2, -1)

        logp_left1 = logp_left1[torch.arange(batch).unsqueeze(1), target]
        logp_left1 = logp_left1[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                                torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        logp_left2 = logp_left2[torch.arange(batch).unsqueeze(1), target]
        logp_left2 = logp_left2[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                                torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        loss_left1_mask = torch.tril(target.new_ones(num, num), -1).unsqueeze(0).expand(batch, num, num)
        truth_left1 = loss_left1_mask - torch.tril(target.new_ones(num, num), -2).unsqueeze(0)

        loss_left2_mask = torch.tril(target.new_ones(num, num), -2).unsqueeze(0).expand(batch, num, num)
        truth_left2 = loss_left2_mask - torch.tril(target.new_ones(num, num), -3).unsqueeze(0)

        loss_left1 = self.critic(logp_left1.view(-1, 2), truth_left1.contiguous().view(-1))
        loss_left2 = self.critic(logp_left2.view(-1, 2), truth_left2.contiguous().view(-1))

        eye_mask = torch.eye(num).byte().cuda().unsqueeze(0)
        rela_mask = torch.ones_like(truth_left1).byte() - eye_mask

        left1_mask = loss_left1_mask.clone()
        left2_mask = loss_left1_mask.clone()

        for b in range(batch):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len_less[b]] = 1

            rela_mask[b, tgt_len[b]:] = 0
            rela_mask[b, :, tgt_len[b]:] = 0

            left1_mask[b, tgt_len[b]:] = 0
            left2_mask[b, tgt_len[b]:] = 0

            if tgt_len[b] >= 4:
                for ix in range(3, tgt_len[b]):
                    weight = document_matrix.new_ones(ix)
                    weight[-1] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left1_mask[b, ix, negix] = 0

                    weight[-1] = 1
                    weight[-2] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left2_mask[b, ix, negix] = 0

        loss_rela.masked_fill_(rela_mask.view(-1) == 0, 0)
        loss_rela = loss_rela.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_rela = loss_rela.sum() / batch

        loss_left1.masked_fill_(left1_mask.view(-1) == 0, 0)
        loss_left1 = loss_left1.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_left1 = loss_left1.sum() / batch

        loss_left2.masked_fill_(left2_mask.view(-1) == 0, 0)
        loss_left2 = loss_left2.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_left2 = loss_left2.sum() / batch

        # B N-1 H
        dec_inputs = document_matrix[torch.arange(document_matrix.size(0)).unsqueeze(1), target[:, :-1]]
        start = dec_inputs.new_zeros(batch, 1, dec_inputs.size(2))
        # B N H
        dec_inputs = torch.cat((start, dec_inputs), 1)

        p_direc = F.softmax(score, -1)
        rela_vec_diret = torch.cat((rela_vec, p_direc), -1)
        p_left1 = F.softmax(score_left1, -1)
        p_left2 = F.softmax(score_left2, -1)

        hist_vec_left1 = torch.cat((rela_hist_left1, p_left1), -1)
        hist_vec_left2 = torch.cat((rela_hist_left2, p_left2), -1)

        dec_outputs = []
        pw_keys = []
        # mask already pointed nodes
        pointed_mask = [rela_mask.new_zeros(batch, 1, num)]

        eye_zeros = torch.ones_like(eye_mask) - eye_mask
        eye_zeros = eye_zeros.unsqueeze(-1)

        for t in range(num):
            if t == 0:
                rela_mask = rela_mask.unsqueeze(-1)
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)
            else:
                # B (left1)
                tar = target[:, t - 1]

                # future
                rela_mask[torch.arange(batch), tar] = 0
                rela_mask[torch.arange(batch), :, tar] = 0

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(batch), :, tar] = 1
                if t > 1:
                    l2_mask[torch.arange(batch), :, target[:, t - 2]] = 1

                pm = pointed_mask[-1].clone().detach()
                pm[torch.arange(batch), :, tar] = 1
                pointed_mask.append(pm)

            # history information
            cur_hist_l1 = hist_vec_left1.masked_fill(l1_mask == 0, 0).sum(2)
            cur_hist_l2 = hist_vec_left2.masked_fill(l2_mask == 0, 0).sum(2)

            # future information
            rela_vec_diret.masked_fill_(rela_mask == 0, 0)
            forw_pw = rela_vec_diret.mean(2)
            back_pw = rela_vec_diret.mean(1)

            pw_info = torch.cat((cur_hist_l1, cur_hist_l2, forw_pw, back_pw), -1)
            pw_key = self.pw_k(pw_info)
            pw_keys.append(pw_key.unsqueeze(1))

            dec_inp = dec_inputs[:, t:t + 1]

            # B 1 H
            output, hcn = self.decoder(dec_inp, hcn)
            dec_outputs.append(output)

        # B N-1 H
        dec_outputs = torch.cat(dec_outputs, 1)
        # B qN 1 H
        query = self.linears[0](dec_outputs).unsqueeze(2)
        # B 1 kN H
        PWkey = torch.cat(pw_keys, 1)
        key = torch.cat((PWkey, GRNkey.unsqueeze(1).expand_as(PWkey)), dim = -1)
        key = self.newkey(key)

        # B qN kN H
        e = torch.tanh(query + key)
        # B qN kN
        e = self.linears[2](e).squeeze(-1)

        # mask already pointed nodes
        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target_less.contiguous().view(-1))

        loss.masked_fill_(target_mask.view(-1) == 0, 0)
        loss = loss.sum() / batch

        total_loss = loss + (loss_rela + loss_left1 + loss_left2) * self.lambda2
        if torch.isnan(total_loss):
            exit('nan')
        return total_loss, loss, loss_rela, loss_left1, loss_left2, document_matrix

    def rnn_enc(self, src_and_len, doc_num):
        '''
        :param src_and_len: src:batch×batch中的最大句子个数，最大词个数 len是每行句子的词数
        :param doc_num: B, each doc has sentences number
        :return: document matirx:batch，最大句子数（每个文档）,512
        '''
        src, length = src_and_len
        sorted_len, ix = torch.sort(length, descending=True)
        sorted_src = src[ix]

        # bi-rnn must uses pack, else needs mask   speedup
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_src, sorted_len, batch_first=True)
        x = packed_x.data

        x = self.src_embed(x)

        if self.emb_dp > 0:
            x = F.dropout(x, self.emb_dp, self.training)
        packed_x = nn.utils.rnn.PackedSequence(x, packed_x.batch_sizes)

        # 2 TN H
        states, (hn, _) = self.sen_enc(packed_x)

        # TN T 2H
        allwordstates, _ = nn.utils.rnn.pad_packed_sequence(states, True)

        # TN 2H
        hn = hn.transpose(0, 1).contiguous().view(src.size(0), -1)

        _, recovered_ix = torch.sort(ix, descending=False)
        hn = hn[recovered_ix]
        allwordstates = allwordstates[recovered_ix]

        batch_size = len(doc_num)
        maxdoclen = max(doc_num)
        output = hn.view(batch_size, maxdoclen, -1)

        allwordstates = allwordstates.view(batch_size, -1, hn.size(-1))
        return output, allwordstates


    def bert_enc(self, src, doc_num, ewords, bert_token, blocs):
        # with torch.no_grad():
            batch_size = len(doc_num)
            max_doc_len = max(doc_num)
            max_sent_len = len(bert_token['input_ids'][0]) - 1
            max_word_len = len(ewords[0])
            output = self.bert(**bert_token)[0]
            attention_mask = bert_token['attention_mask'].float()
            cls_mask = torch.cat((torch.zeros(attention_mask.size(0),1),torch.ones(attention_mask.size(0),attention_mask.size(1)-1)),dim=-1).cuda()
            attention_mask = attention_mask*cls_mask

            attention_mask = attention_mask/attention_mask.sum(-1).unsqueeze(-1)

            sentences = (output*attention_mask.unsqueeze(-1)).sum(-2)
            sentences = sentences.view(batch_size, max_doc_len, -1)
            cls_words = output.split([1, max_sent_len], dim=1)
            #sentences = cls_words[0].view(batch_size, max_doc_len, -1)

            allwords = cls_words[1].split(max_doc_len, dim=0)
            if ewords.size(1) == 0:
                entities = torch.zeros(batch_size, 0, 768).cuda()
                return sentences, entities

            entities = []
            for batch_index in range(len(ewords)):  # 32
                word_bert_locs = blocs[batch_index]
                batch_word = []
                for word_allsent_loc in word_bert_locs:
                    each_word=[]
                    for word_eachsent_loc in word_allsent_loc:
                        sent_index, locs = word_eachsent_loc
                        word_each_sent = []
                        for loc in locs:
                            word = []
                            for l in loc:
                                word.append(allwords[batch_index][sent_index, l - 1])
                            word = torch.stack(word).mean(-2)
                            word_each_sent.append(word)
                        word_each_sent = torch.stack(word_each_sent).mean(-2)
                        each_word.append(word_each_sent)
                    each_word = torch.stack(each_word).mean(-2)
                    batch_word.append(each_word)
                real_len = len(batch_word)
                for i in range(max_word_len-real_len):
                    batch_word.append(torch.zeros(768).cuda())
                batch_word = torch.stack(batch_word)
                entities.append(batch_word)
            entities = torch.stack(entities)
            return sentences, entities

    def encode(self, src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, ecom, blocs):
        # get sentence emb and mask#######################
        # sentences, words_states = self.rnn_enc(src_and_len, doc_num)
        sentences, words_states = self.bert_enc(src_and_len[0], doc_num, ewords_and_len[0], bert_token, blocs)

        if self.model_dp > 0:
            sentences = F.dropout(sentences, self.model_dp, self.training)

        batch = sentences.size(0)
        sents_mask = sentences.new_zeros(batch, sentences.size(1)).byte()

        for i in range(batch):
            sents_mask[i, :doc_num[i]] = 1

        sentences.masked_fill_(sents_mask.unsqueeze(2) == 0, 0)

        # get entity emb and mask
        words, _ = ewords_and_len
        # <pad> 1
        words_mask = (words != 1)

        #entity_emb = self.src_embed(words)
        if self.emb_dp > 0:
            words_states = F.dropout(words_states, self.emb_dp, self.training)

        para, hn = self.encoder(sentences, sents_mask, words_states, words_mask, elocs, slocs, ecom)

        hn = hn.unsqueeze(0)
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)

        keyinput = torch.cat((sentences, para), -1)
        key = self.linears[1](keyinput)

        return sentences, para, hcn, key

    def step(self, prev_y, prev_handc, keys, mask):
        '''
        :param prev_y: (seq_len=B, 1, H)
        :param prev_handc: (1, B, H)
        :return:
        '''
        # 1 B H
        _, (h, c) = self.decoder(prev_y, prev_handc)
        # 1 B H-> B H-> B 1 H
        query = h.squeeze(0).unsqueeze(1)
        query = self.linears[0](query)
        # B N H
        e = torch.tanh(query + keys)
        # B N
        e = self.linears[2](e).squeeze(2)
        '''
        keys = keys.transpose(1, 2)
        e = torch.matmul(query, keys).squeeze(1)
        '''
        mask = (mask != 0)
        e.masked_fill_(mask, -1e9)
        logp = F.log_softmax(e, dim=-1)

        return h, c, logp

    def rela_att(self, prev_h, rela, rela_k, rela_mask):
        # B 1 H
        q = self.rela_q(prev_h).transpose(0, 1)
        e = self.rela_e(torch.tanh(q + rela_k))

        e.masked_fill_(rela_mask == 0, -1e9)
        alpha = F.softmax(e, 1)
        context = torch.sum(alpha * rela, 1, True)
        return context

    def stepv2(self, prev_y, prev_handc, GRNkeys, mask, rela_vec, hist_left1, hist_left2, rela_mask, l1_mask, l2_mask):
        '''
        :param prev_y: (seq_len=B, 1, H)
        :param prev_handc: (1, B, H)
        :return:
        '''

        _, (h, c) = self.decoder(prev_y, prev_handc)
        # 1 B H-> B H-> B 1 H
        query = h.squeeze(0).unsqueeze(1)
        query = self.linears[0](query)

        # history
        left1 = hist_left1.masked_fill(l1_mask.unsqueeze(-1) == 0, 0).sum(2)
        left2 = hist_left2.masked_fill(l2_mask.unsqueeze(-1) == 0, 0).sum(2)

        # future
        rela_vec.masked_fill_(rela_mask.unsqueeze(-1) == 0, 0)
        forw_futu = rela_vec.mean(2)
        back_futu = rela_vec.mean(1)

        pw = torch.cat((left1, left2, forw_futu, back_futu), -1)
        PWkeys = self.pw_k(pw)

        keys = torch.cat((PWkeys, GRNkeys.expand_as(PWkeys)), dim =-1)
        keys = self.newkey(keys)

        # B N H
        e = torch.tanh(query + keys)
        # B N
        e = self.linears[2](e).squeeze(2)
        e.masked_fill_(mask, -1e9)

        logp = F.log_softmax(e, dim=-1)
        return h, c, logp

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False).cuda()
        # self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False)

def beam_search_pointer(args, model, src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, ecom, blocs):
    sentences, GRN_sents, dec_init, GRNkeys = model.encode(src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, ecom, blocs)
    document = sentences.squeeze(0)
    T, H = document.size()

    # future
    rela_out, rela_vec, hist_left1, hist_left2 = model.rela_pred(sentences, dec_init[0])

    eye_mask = torch.eye(T).cuda().byte()
    eye_zeros = torch.ones_like(eye_mask) - eye_mask

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()

            rela_mask = eye_zeros.unsqueeze(0)

            l1_mask = torch.zeros_like(rela_mask)
            l2_mask = torch.zeros_like(rela_mask)
        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()
            # beam 1 H
            dec_input = document[index].unsqueeze(1)

            temp_batch = index.size(0)

            pointed_mask[torch.arange(temp_batch), index] = 1

            rela_mask[torch.arange(temp_batch), :, index] = 0
            rela_mask[torch.arange(temp_batch), index] = 0

            l1_mask = torch.zeros_like(rela_mask)
            l2_mask = torch.zeros_like(rela_mask)

            l1_mask[torch.arange(temp_batch), :, index] = 1
            if t > 1:
                left2_index = index.new_tensor(list(map(lambda cand: cand[-2], candidates)))
                l2_mask[torch.arange(temp_batch), :, left2_index] = 1

        dec_h, dec_c, log_prob = model.stepv2(dec_input, dec_init, GRNkeys, pointed_mask.bool(),
                                              rela_vec, hist_left1, hist_left2, rela_mask, l1_mask, l2_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src_and_len[0].new_tensor(remain_list)

        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        rela_mask = rela_mask.index_select(0, beam_remain_ix)
        rela_vec = rela_vec.index_select(0, beam_remain_ix)

        hist_left1 = hist_left1.index_select(0, beam_remain_ix)
        hist_left2 = hist_left2.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output, rela_out


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))


# train(args, train_real, dev_real, (DOC, ORDER, GRAPH), checkpoint)
# Namespace(agg='gate', alpha=0.6, batch_size=2, beam_size=64,
# corpus=('aan/train.lower', 'aan/train.eg.20'), d_emb=100, d_mlp=500, d_rnn=500,
# decoding_path='decoding', delay=1, doc_vocab=19279, drop_ratio=0.5, early_stop=5, ehid=150, entityemb='glove',
# eval_every=100, gnndp=0.3, gnnl=3, grad_clip=0.0, initnn='standard', input_drop_ratio=0.5, keep_cpts=1, labeldim=50,
# lang=None, length_ratio=2, load_from=None, load_vocab=False, loss=0, lr=1.0, lrdecay=0, main_path='./',
# max_len=None, maximum_steps=100, mode='train', model='07.21_10.01.', model_path='models', n_heads=2, n_layers=5,
# optimizer='Noam', params='user', patience=0, pool=100, ref=None, reglamb=0, resume=False, save_every=50, seed=1234,
# senenc='bow', share_embed=False, share_vocab=False, smoothing=0.0, test=('aan/test.lower', 'aan/test.eg.20'),
#
# valid=('aan/val.lower', 'aan/val.eg.20'), vocab='aan/vocab.new.100d.lower.pt', vocab_size=40000, warmup=4000, writetrans='decoding/ann_0.5_gdp_0.3_gl3.devorder')
def train(args, train_iter, dev, fields, checkpoint, teacher=None):
    model = PointerNet(args)
    model.cuda()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    if torch.cuda.device_count() > 1:
        for i,layer in enumerate(model.bert.encoder.layer):
            if i > 2 :
                layer.to('cuda:1')
            if i > 5 :
                layer.to('cuda:2')
            if i > 8 :
                layer.to('cuda:3')

    #for para in model.bert.parameters():
    #    para.requires_grad = False
    
    bert_params = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    DOC, ORDER, GRAPH = fields
    # print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    wd = 1e-5
    # opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)
    opt = torch.optim.Adadelta([{'params': filter(lambda p: p.requires_grad, model.bert.parameters()), 'lr': args.lr},
                               {'params': base_params}], lr=1.0, rho=0.95, weight_decay=wd)
    # opt = torch.optim.Adadelta(base_params, lr=1.0, rho=0.95, weight_decay=wd)

    best_score = -np.inf
    best_iter = 0
    offset = 0

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()

    early_stop = args.early_stop

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH)
    test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    for epc in range(args.maximum_steps):
        model.zero_grad()
        for iters, batch in enumerate(train_iter):
            model.train()
            t1 = time.time()
            loss, point_loss, rela_loss, left1_loss, left2_loss, sentences = model(batch.doc, batch.order,
                                                                                   batch.doc_len,
                                                                                   batch.e_words, batch.elocs,
                                                                                   batch.slocs, batch.token, batch.ecom, batch.blocs)

            accu_loss = loss/args.accumulation_steps
            accu_loss.backward()
            if (iters+1) % args.accumulation_steps ==0:
                t2 = time.time()
                print('epc:{} iter:{} loss:{:.2f} futu:{:.2f} left1:{:.2f}, left2:{:.2f} total_loss:{:.2f} \
                t:{:.2f} lr:{:.1e}'.format(epc, (iters + 1)/args.accumulation_steps, point_loss, rela_loss, left1_loss, left2_loss,
                                       accu_loss,
                                       t2 - t1, opt.param_groups[0]['lr']))
                opt.step()
                opt.zero_grad()
                model.zero_grad()
        if (iters+1) % args.accumulation_steps !=0:
            opt.step()
            opt.zero_grad()
            model.zero_grad()


        if epc < 0:
            continue

        with torch.no_grad():
            print('valid..............')
            if args.loss:
                score = valid_model(args, model, dev, DOC, 'loss')
                print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, score, best_score))
            else:
                score, pmr, ktau, _ = valid_model(args, model, dev, DOC)
                print('epc:{}, val acc:{:.4f} best:{:.4f} pmr:{:.2f} ktau:{:.4f}'.format(epc, score, best_score,
                                                                                         pmr, ktau))

            if score > best_score:
                best_score = score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'args': args,
                              'loss': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2f}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2f}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cuda')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        acc, pmr, ktau, pm = valid_model(args, model, test_real, DOC, shuflle_times=1)
        print('test acc:{:.4%} pmr:{:.2%} ktau:{:.4f} pm:{:.2%}'.format(acc, pmr, ktau, pm))


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0

        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,
                         dev_batch.slocs, dev_batch.token, dev_batch.ecom,dev_batch.blocs)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        f = open(args.writetrans, 'w')

        if args.beam_size != 1:
            print('beam search with beam', args.beam_size)

        best_acc = []
        for epc in range(shuflle_times):
            truth = []
            predicted = []

            for j, dev_batch in enumerate(dev):
                tru = dev_batch.order[0].view(-1).tolist()
                truth.append(tru)

                if len(tru) == 1:
                    pred = tru
                else:

                    pred, out = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len, dev_batch.e_words,
                                                    dev_batch.elocs, dev_batch.slocs, dev_batch.token, dev_batch.ecom,dev_batch.blocs)

                predicted.append(pred)
                print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                      file=f)

            right, total = 0, 0
            pmr_right = 0
            taus = []
            # pm
            pm_p, pm_r = [], []
            import itertools

            from sklearn.metrics import accuracy_score

            for t, p in zip(truth, predicted):
                if len(p) == 1:
                    right += 1
                    total += 1
                    pmr_right += 1
                    taus.append(1)
                    continue

                eq = np.equal(t, p)
                right += eq.sum()
                total += len(t)

                pmr_right += eq.all()

                # pm
                s_t = set([i for i in itertools.combinations(t, 2)])
                s_p = set([i for i in itertools.combinations(p, 2)])
                pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
                pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

                cn_2 = len(p) * (len(p) - 1) / 2
                pairs = len(s_p) - len(s_p.intersection(s_t))
                tau = 1 - 2 * pairs / cn_2
                taus.append(tau)

            # acc = right / total

            acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                                 list(itertools.chain.from_iterable(predicted)))

            best_acc.append(acc)

            pmr = pmr_right / len(truth)
            taus = np.mean(taus)

            pm_p = np.mean(pm_p)
            pm_r = np.mean(pm_r)
            pm = 2 * pm_p * pm_r / (pm_p + pm_r)

            print('acc:', acc)

        f.close()
        acc = max(best_acc)
        return acc, pmr, taus, pm


def decode(args, test_real, fields, checkpoint):
    with torch.no_grad():
        model = PointerNet(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])
        DOC, ORDER = fields
        acc, pmr, ktau, _ = valid_model(args, model, test_real, DOC)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2%}'.format(acc, pmr, ktau))
