
DATA_PATH=/data/you/data/XLM/multiun/




export NGPU=2;
 CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=$NGPU  train.py \
--exp_name mulitun --exp_id  multiUN_mnmt  \
--enc_layers 6 \
--data_path $DATA_PATH \
--dec_layers 6 \
--emb_dim 1024 \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  \
--encoder_only False \
--mt_step 'ar-en,ar-es,ar-ru,en-ar,en-es,en-ru,es-ar,es-en,es-ru,ru-ar,ru-en,ru-es' \
--n_layers 6 \
--n_heads 8 \
--dropout '0.1' \
--bptt 256 \
--tokens_per_batch 2400  \
--max_vocab 95000 \
--optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' \
--epoch_size 100000 \
--eval_bleu True \
--stopping_criterion 'valid_ar-ru_mt_bleu,100' \
--validation_metrics 'valid_ar-ru_mt_bleu' \
--eval_type valid \
--eval_num -1 \
--beam_size 1 \
--zero_shot ar es ru \
--gelu_activation False  \
--enc_attention_dropout 0 \
--dec_attention_dropout 0  \
--dec_langemb False \
--enc_langemb False \
--norm_emb False \
--lang_specid "ar:7,en:8,es:9,fr:10,ru:11,zh:12"  --dec_special True   \
--master_port 36110 --debug_train True

