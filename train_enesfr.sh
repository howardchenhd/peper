#!/usr/bin/env bash
export NGPU=1;

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=$NGPU  train.py \
--dump_path /data/bjji/github/model/eurparl --exp_name tlm --exp_id  en_es_fr \
--enc_layers 6 \
--data_path '/data/bjji/github/data/en_es_fr' \
--dec_layers 6 \
--emb_dim 1024 \
--lgs 'en-es-fr' \
--encoder_only False \
--mt_step 'en-es,fr-es' \
--n_layers 6 \
--n_heads 8 \
--dropout '0.1' \
--bptt 256 \
--tokens_per_batch 2500 \
--max_vocab 95000 \
--optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' \
--epoch_size 500000 \
--eval_bleu True \
--stopping_criterion 'valid_fr-es_mt_bleu,100' \
--validation_metrics 'valid_fr-es_mt_bleu' \
--eval_type valid \
--eval_num -1 \
--save_periodic 1 \
--beam_size 1 \
--fix_enc True \
--zero_shot fr es  \
--reset_lang "en:0,fr:1,es:0"  \
--enc_langnum 2 \
--fix_enc True \
--gelu_activation true  --attention_dropout 0.1   \
--reload_model "/data/bjji/github/model/eurparl/tlm_base/tlm_en_fr_eurp_half/best-valid_mlm_ppl.pth,"
