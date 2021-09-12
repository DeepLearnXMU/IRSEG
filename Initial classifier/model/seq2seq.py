import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from random import shuffle
from transformers import AutoModel
from model.generator import Beam
from datatool.data import DocField, DocDataset, DocIter


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

        self.e_gru = GRUCell(e_emb + sh_dim + label_dim + g_dim, eh_dim)  # 实体编码、上图层句消息、标签、上图层全局隐向量

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

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    # para, hn = self.encoder(sentences, sents_mask, entity_emb, words_mask, elocs)
    def forward(self, sent, smask, word, wmask, elocs, slocs):
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
        '''
        # *********OLD s2smatrix ***********
        s2smatrix = torch.matmul(mask_se, mask_se_t)
        s2smatrix = s2smatrix != 0
        eye = torch.eye(snum).byte().cuda()
        s2s = smask.new_ones(snum, snum)
        eyemask = (s2s - eye).unsqueeze(0)

        s2smatrix = s2smatrix * eyemask
        s2smatrix = s2smatrix & smask.unsqueeze(1)
        s2smatrix = s2smatrix.float()
        '''

        # *********NEW s2sorder ***********
        target = torch.zeros(batch, snum, snum).float().cuda()
        s2sorder = torch.zeros(batch, snum, snum).float().cuda()
        for ib, sloc in enumerate(slocs):
            for triple in sloc:
                target[ib, triple[0], triple[1]] = 1.
                s2sorder[ib, triple[0], triple[1]] = triple[2]
                s2sorder[ib, triple[1], triple[0]] = 1. - triple[2]
        s2smask = target + target.transpose(-1,-2)
        
        s_h = torch.zeros_like(sent)
        # print("s_h size:",s_h.size())
        g_h = sent.new_zeros(batch, self.s_hid)
        # print("g_h size:",g_h.size())
        e_h = sent.new_zeros(batch, wnum, self.e_hid)
        # print("e_h size:",e_h.size())

        for i in range(self.layer):
            # 1.aggregation
            # s_neigh_s_h = self.mean(s_h, s2sorder)
            s_neigh_s_h = self.sum(s_h, s2sorder)

            # B S E H
            if self.agg == 'gate':
                # 句对句间
                '''
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, snum, self.s_hid)
                s_h_expand_t = s_h.unsqueeze(1).expand(batch, snum, snum, self.s_hid)

                s_h_expand_order = torch.cat((s_h_expand_t, order_emb), -1)
                s_h_expand_order_value = torch.cat((s_h_expand_order, value_emb), -1)
                s_s_l = torch.cat((s_h_expand, s_h_expand_order_value), -1)

                gs = torch.sigmoid(self.gate0(s_s_l))
                # s_neigh_s_h = s_h_expand_order_value * gs * s2smask.unsqueeze(3)
                s_neigh_s_h = s_h_expand_t * gs * s2smask.unsqueeze(3)
                s_neigh_s_h = s_neigh_s_h.sum(2)
                '''

                # 句对词
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, wnum, self.s_hid)
                # print("s_h_expand size:",s_h_expand.size())
                e_h_expand = e_h.unsqueeze(1).expand(batch, snum, wnum, self.e_hid)
                # print("e_h_expand size:",e_h_expand.size())

                # 带上了label信息的实体的隐状态
                e_h_expand_edge = torch.cat((e_h_expand, label_emb), -1)
                # print("e_h_expand_edge:",e_h_expand_edge.size())

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

            s_input = torch.cat((sent, s_neigh_s_h, s_neigh_e_h), -1)
            e_input = torch.cat((word, e_neigh_s_h), -1)

            # 2.update
            s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))

        if self.dp > 0:
            s_h = F.dropout(s_h, self.dp, self.training)

        return (s_h, s2smask, target), g_h


class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio
        self.d_emb = args.d_emb
        self.sen_enc_type = args.senenc

        self.src_embed = nn.Embedding(args.doc_vocab, 100)  # 把document的词汇编码成embedding

        #self.bert = AutoModel.from_pretrained("bert-base-uncased").to('cuda')
        self.bert=None

        self.sen_enc = nn.LSTM(self.d_emb, args.d_rnn // 2, bidirectional=True,
                               batch_first=True)  # 把document的词汇的embedding编码为句子级别的向量

        self.entityemb = args.entityemb

        self.encoder = GRNGOB(s_emb=args.d_rnn,
                              e_emb=args.d_emb if self.entityemb == 'glove' else args.d_rnn,
                              s_hidden=args.d_rnn,
                              e_hidden=args.ehid, label_dim=args.labeldim,
                              layer=args.gnnl, dp=args.gnndp, agg=args.agg)

        d_mlp = args.d_mlp

        self.order_predictor = OrderPredictor(d_mlp, args.d_rnn)

        self.linears = nn.ModuleList([nn.Linear(args.d_rnn, d_mlp),
                                      nn.Linear(args.d_rnn * 2, d_mlp),
                                      nn.Linear(d_mlp, 1)])
        self.decoder = nn.LSTM(args.d_rnn, args.d_rnn, batch_first=True)
        self.critic = None

    def equip(self, critic):
        self.critic = critic

    def forward(self, src_and_len, tgt_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, blocs=None):

        document_matrix, para, hcn, key = self.encode(src_and_len, doc_num, ewords_and_len, elocs, slocs,
                                                      bert_token, blocs)
        s_h, s2smask, target = para
        s2smatrix = target+target.transpose(-1, -2)
        batch = s_h.size(0)
        snum = s_h.size(1)
        s_h_expand = s_h.unsqueeze(2).expand(batch, snum, snum, self.encoder.s_hid)
        s_h_expand_t = s_h.unsqueeze(1).expand(batch, snum, snum, self.encoder.s_hid)
        s_nei_s = torch.cat((s_h_expand, s_h_expand_t), dim=-1)
        input = s_nei_s[s2smask == 1]
        order = target[s2smask == 1]
        
        # order_wo_noise = target[(s2smatrix-s2smask) == 1]
        order_prob = self.order_predictor(input).squeeze()


        loss_func = nn.BCELoss()
        loss = loss_func(order_prob, order)

        if torch.isnan(loss):
            exit('nan')
        return loss

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
        batch_size = len(doc_num)
        max_doc_len = max(doc_num)
        max_sent_len = len(bert_token['input_ids'][0]) - 1
        max_word_len = len(ewords[0])
        output = self.bert(**bert_token)
        cls_words = output[0].split([1, max_sent_len], dim=1)
        sentences = cls_words[0].view(batch_size, max_doc_len, -1)
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

    def pre_procces(self, src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, blocs=None):
        # get sentence emb and mask
        sentences, words_states = self.rnn_enc(src_and_len, doc_num)
        # sentences, words_states = self.bert_enc(src_and_len[0], doc_num, ewords_and_len[0], bert_token, blocs)

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

        entity_emb = self.src_embed(words)

        if self.emb_dp > 0:
            words_states = F.dropout(entity_emb, self.emb_dp, self.training)
        return sentences, sents_mask, words_states, words_mask

    def enc(self, sentences, sents_mask, words_states, words_mask, elocs, slocs):
        para, hn = self.encoder(sentences, sents_mask, words_states, words_mask, elocs, slocs)
        s_h, s2smask, target = para
        hn = hn.unsqueeze(0)
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)

        keyinput = torch.cat((sentences, s_h), -1)
        key = self.linears[1](keyinput)
        return para, hcn, key

    def encode(self, src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, blocs=None):
        sentences, sents_mask, words_states, words_mask = self.pre_procces(src_and_len, doc_num, ewords_and_len, elocs,
                                                                           slocs, bert_token, blocs)

        para, hcn, key = self.enc(sentences, sents_mask, words_states, words_mask, elocs, slocs)
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

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False).cuda()
        # self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False)

    def order_predict(self, src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, blocs=None, f=None):

        sentences, para, hcn, key = self.encode(src_and_len, doc_num, ewords_and_len, elocs, slocs, bert_token, blocs)
        s_h, s2smask, target = para

        batch = s_h.size(0)
        snum = s_h.size(1)

        s_h_expand = s_h.unsqueeze(2).expand(batch, snum, snum, self.encoder.s_hid)
        s_h_expand_t = s_h.unsqueeze(1).expand(batch, snum, snum, self.encoder.s_hid)
        s_nei_s = torch.cat((s_h_expand, s_h_expand_t), dim=-1)
        input = s_nei_s[s2smask == 1]
        order_prob = self.order_predictor(input)
        
        total_order = torch.zeros(batch,snum,snum).float().cuda()
        total_order[s2smask==1] = order_prob.squeeze()
        total_order[s2smask==1] = order_prob.squeeze()/(total_order+total_order.transpose(-1,-2))[s2smask==1]
        self.log_slocs(slocs,total_order, f)
        return order_prob, target[s2smask == 1]

    def log_slocs(self, slocs, order_prob, file):
        
        for ib, sloc in enumerate(slocs):
            for triple in sloc:
                file.write(str(order_prob[ib, triple[0], triple[1]].tolist())+' ')
        file.write('\n')

    def update_slocs(self, slocs, order_prob):
        newslocs = []
        for ib, sloc in enumerate(slocs):
            temp = []
            for triple in sloc:
                temp.append((triple[0], triple[1], order_prob[ib, triple[0], triple[1]]))

            newslocs.append(temp)
        return newslocs


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
def train(args, train_iter, dev, fields, checkpoint):
    model = PointerNet(args)
    model.cuda()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    '''
    #for para in model.bert.embeddings.parameters():
    for para in model.bert.parameters():
        para.requires_grad = False

    for layer in range(9,12):
       for para in model.bert.encoder.layer[layer].parameters():
          para.requires_grad = True
    '''
    #bert_params = list(map(id, model.bert.parameters()))
    #base_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    DOC, ORDER, GRAPH = fields
    # print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    wd = 1e-5
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)
    # opt = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, rho=0.95, weight_decay=wd)
    #opt = torch.optim.Adadelta([
    #    {'params': filter(lambda p: p.requires_grad, model.bert.parameters()), 'lr': 3e-3},
    #    {'params': base_params}], lr=args.lr, rho=0.95, weight_decay=wd)

    best_score = -np.inf
    best_iter = 0

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()

    early_stop = args.early_stop

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH)
    test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    for epc in range(args.maximum_steps):
        for iters, batch in enumerate(train_iter):
            model.train()

            model.zero_grad()

            t1 = time.time()
            loss_print = model(batch.doc, batch.order, batch.doc_len, batch.e_words, batch.elocs, batch.slocs,
                               batch.token, batch.blocs)
            loss = loss_print
            loss.backward()
            opt.step()

            t2 = time.time()
            if iters % 1 == 0:
                print('epc:{} iter:{} loss:{:.4f} t:{:.3f} lr:{:.1e}'.format(epc, iters + 1, loss, t2 - t1,
                                                                             opt.param_groups[0]['lr']))
        if epc < 5:
            continue

        with torch.no_grad():
            print('valid..............')
            if args.loss:
                score = order_model(args, model, dev, 'loss')
                print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, score, best_score))
            else:
                score = order_model(args, model, dev)
                print('epc:{}, val acc:{:.4f} best:{:.4f} '.format(epc, score, best_score))

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
        acc = order_model(args, model, test_real, shuflle_times=1)
        print('epc:{}, val Acc:{:.4f} best:{:.4f} '.format(epc, acc, best_score))


def order_model(args, model, dev, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0
        print('loss')
        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,
                         dev_batch.slocs, dev_batch.token)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        f = open(args.writetrans, 'w', encoding='utf-8')
        f2 = open('{}.slocs'.format(args.model), 'w', encoding='utf-8')
        truth = []
        predicted = []

        for j, dev_batch in enumerate(dev):
            if len(dev_batch.slocs[0]) < 1:
                print('', file=f)
                continue

            logit, tru = model.order_predict(dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,
                                             dev_batch.slocs, dev_batch.token, dev_batch.blocs, f=f2)
            truth.append(tru.tolist())
            print('{}'.format(' '.join(map(str, logit.tolist()))), file=f)

            logit[logit > 0.5] = 1.
            logit[logit <= 0.5] = 0.

            predicted.append(logit.tolist())

        # print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
        #      file=f)

        import itertools

        from sklearn.metrics import accuracy_score

        # acc = right / total
        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                             list(itertools.chain.from_iterable(predicted)))

        print('Acc:', acc)
        f.close()
        f2.close()
        return acc


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0

        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,
                         dev_batch.slocs, dev_batch.token)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        

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
                    pred = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len, dev_batch.e_words,
                                               dev_batch.elocs, dev_batch.slocs, dev_batch.token)

                predicted.append(pred)

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
        acc = order_model(args, model, test_real, DOC)
        print('test acc:{}'.format(acc))
