
DATA_BIN='/data/you/data/ldc_xlm_bin/'
export NGPU=4; 
CUDA_VISIBLE_DEVICES=2,3,4,5    python -m torch.distributed.launch --nproc_per_node=$NGPU  train.py \
--exp_name ldc   \
--enc_layers 6 \
--data_path $DATA_BIN \
--dec_layers 6 \
--emb_dim 512 \
--lgs 'ch-en' \
--encoder_only False \
--mt_step 'ch-en' \
--n_layers 6 \
--n_heads 8 \
--bptt 256 \
--tokens_per_batch 4000 \
--optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --label_smooth 0.1 \
--epoch_size 100000 \
--eval_bleu True \
--stopping_criterion 'valid_ch-en_mt_bleu,100'  \
--validation_metrics 'valid_ch-en_mt_bleu' \
--eval_type valid  \
--eval_num -1 \
--beam_size 5 \
--fix_enc False \
--dropout '0.1' \
--enc_attention_dropout 0 \
--dec_attention_dropout 0 \
--master_port 30123 \
--enc_langemb False \
--dec_langemb False \
--gelu_activation False 



