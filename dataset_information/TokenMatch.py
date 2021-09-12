from transformers import AutoTokenizer

def search(entity:str,sent_tok:list):
    etok_locs=[]
    for index in range(len(sent_tok)):
        sub_words = []
        if entity == sent_tok[index]:
            sub_words.append(index)
            etok_locs.append(sub_words)
        elif entity.startswith(sent_tok[index]):
            sub_word=sent_tok[index]
            sub_words.append(index)
            for j in range(index+1,len(sent_tok)):
                if sent_tok[j].startswith('##'):
                    sub_word = sub_word+sent_tok[j].lstrip('##')
                else:
                    sub_word = sub_word+sent_tok[j]
                sub_words.append(j)
                if entity==sub_word:
                    etok_locs.append(sub_words)
                    break
                if len(sub_word)>len(entity):
                    break

    return etok_locs
def match(mode:str,data:str):
    textfile = open('{}/{}.lower'.format(data,mode),'r',encoding='utf-8')
    enfile = open('{}/{}.eg.20.nosingle'.format(data,mode),'r',encoding='utf-8')
    textlines =textfile.readlines()
    enlines = enfile.readlines()
    textfile.close()
    enfile.close()
    outfile = open('{}/{}.en.bertlocs'.format(data,mode),'w',encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for index in range(len(textlines)):
        paragraph = textlines[index].rstrip('\n').split('<eos>')
        elocs =[]
        enline = enlines[index].rstrip('\n')
        if len(enline)<1:
            outfile.write('\n')
            continue
        for en_roles in enline.split(' '):
            en,roles = en_roles.split(':')
            locs=[]
            for role in roles.split('|'):
                locs.append(int(role.split('-')[0]))
            elocs.append((en,locs))
        paragraph_ids = tokenizer(paragraph, return_tensors='pt', padding=True)['input_ids']
        paragraph_tokens=[]
        for ids in paragraph_ids:
            paragraph_tokens.append(tokenizer.convert_ids_to_tokens(ids))

        etok_locs=[]
        outstr=''
        for eloc in elocs:
            en, locs = eloc
            enstr = '{}:'.format(en)
            tok_locs =[]
            for loc in locs:
                tok_locs_for_each_sent = search(en,paragraph_tokens[loc])
                tok_locs.append((loc,tok_locs_for_each_sent))
                enstr = '{}{}-{}|'.format(enstr,loc,tok_locs_for_each_sent)
            enstr = enstr.rstrip('|')+'&'
            outstr = outstr+enstr
            etok_locs.append((en,tok_locs))
        outstr = outstr.rstrip('&')
        outfile.write(outstr+'\n')
    outfile.close()
    
if __name__ == '__main__':

    data = 'duc'
    #match('train',data)
    match('test',data)
    #match('val',data)
