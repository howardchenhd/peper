#!/usr/bin/env bash
#
#export NGPU=4;
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPU  train.py \
#--exp_name test --exp_id  multiUN_multilinual_norm \
#--enc_layers 6 \
#--data_path '/data4/bjji/data/XLM/multiUN/bin' \
#--dec_layers 6 \
#--emb_dim 1024 \
#--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  \
#--encoder_only False \
#--mt_step 'ar-zh,es-zh,fr-zh,ru-zh,ar-en,en-ar,es-en,en-es,fr-en,en-fr,ru-en,en-ru,en-zh,zh-en' \
#--n_layers 6 \
#--n_heads 8 \
#--dropout '0.1' \
#--bptt 256 \
#--tokens_per_batch 2300 \
#--max_vocab 95000 \
#--optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' \
#--epoch_size 100000 \
#--eval_bleu True \
#--stopping_criterion 'valid_es-zh_mt_bleu,100' \
#--validation_metrics 'valid_es-zh_mt_bleu' \
#--eval_type valid \
#--eval_num 200 \
#--beam_size 1 \
#--fix_enc True \
#--zero_shot ar es fr ru zh \
#--fix_enc True \
#--gelu_activation true  --attention_dropout 0.1   \
#--reload_model "/data4/bjji/data/XLM/europarl/model/xlni/mlm_tlm_xnli15_1024.pth," \
#--norm_emb True



export NGPU=4;

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU  train.py \
--exp_name test --exp_id  multiUN_multilinual_norm_langemb \
--enc_layers 6 \
--data_path '/data4/bjji/data/XLM/multiUN/bin' \
--dec_layers 6 \
--emb_dim 1024 \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  \
--encoder_only False \
--mt_step 'ar-zh,es-zh,fr-zh,ru-zh,ar-en,en-ar,es-en,en-es,fr-en,en-fr,ru-en,en-ru,en-zh,zh-en' \
--n_layers 6 \
--n_heads 8 \
--dropout '0.1' \
--bptt 256 \
--tokens_per_batch 2300 \
--max_vocab 95000 \
--optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' \
--epoch_size 100000 \
--eval_bleu True \
--stopping_criterion 'valid_es-zh_mt_bleu,100' \
--validation_metrics 'valid_es-zh_mt_bleu' \
--eval_type valid \
--eval_num 200 \
--beam_size 1 \
--fix_enc True \
--zero_shot ar es fr ru zh \
--fix_enc True \
--gelu_activation true  --attention_dropout 0.1   \
--reload_model "/data4/bjji/data/XLM/europarl/model/xlni/mlm_tlm_xnli15_1024.pth," \
--norm_emb True \
--lang_emb True \
--master_port 35000
