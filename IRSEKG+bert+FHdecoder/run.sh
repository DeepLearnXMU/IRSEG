#!/bin/bash
datapath=#datapath
batch=#batch_size
accumulation_steps=#acc_steps
lr=#lr

h=768
eh=150
labeldim=50


rand=#random_seed

dp=0.5
mdp=0.5

ld2=0.5
agg=gate

#nos=''
nos=.nosingle

gnnl=3
gnndp=0.3

modelname=#model_name

prefix=lower
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model ${modelname} --vocab $datapath/vocab.new.100d.${prefix}.pt \
    --lambda1 1  --lambda2 ${ld2} \
    --load_from models/${modelname}.best.pt \
    --corpus ../dataset/$datapath/del/train.${prefix} ../dataset/$datapath/train.eg.20${nos} ../dataset/$datapath/#order_file ../dataset/$datapath/#entity-edge-file ../dataset/$datapath/#entity_bert_location_file \
	--valid ../dataset/$datapath/del/val.${prefix} ../dataset/$datapath/del/val.eg.20${nos} ../dataset/$datapath/#order_file ../dataset/$datapath/#entity-edge-file ../dataset/$datapath/#entity_bert_location_file \
    --test ../dataset/$datapath/del/test.${prefix} ../dataset/$datapath/del/test.eg.20${nos} ../dataset/$datapath/#order_file ../dataset/$datapath/#entity-edge-file ../dataset/$datapath/#entity_bert_location_file \
	--loss 0 \
    --writetrans decoding/${modelname}.dev.order --ehid ${eh} --entityemb glove \
    --gnnl ${gnnl} --labeldim ${labeldim} --agg ${agg} --gnndp ${gnndp} \
    --batch_size ${batch} --accumulation_steps ${accumulation_steps} --beam_size 64 --lr ${lr} --seed ${rand} \
    --d_emb 768 --d_rnn ${h} --d_mlp ${h} --d_pair 256 --input_drop_ratio ${dp} --drop_ratio ${mdp} \
    --save_every 50 --maximum_steps 100 --early_stop 5 >${modelname}.try 2>&1 &
