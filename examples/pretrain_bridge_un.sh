
DATA_PATH=/data/you/data/XLM/multiun/
MLM_MODEL=/data/you/data/XLM/europarl/model/mlm_base/mlm_xnli15_1024.pth

export NGPU=2;
CUDA_VISIBLE_DEVICES=3,4    python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name pretrain_bridge  \
--enc_layers 6 \
--data_path $DATA_PATH \
--dec_layers 6 \
--emb_dim 1024 \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh' \
--bridge_steps 'ar-en,en-ar,ru-en,en-ru,en-es,es-en' \
--encoder_only True \
--n_layers 6 \
 --batch_size 25 \
 --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' \
--epoch_size 100000  --max_vocab 95000  \
--eval_bleu False \
--stopping_criterion 'valid_ar-en_mlm_acc,10' \
--validation_metrics 'valid_ar-en_mlm_acc' \
--eval_type valid \
--eval_num 1500 \
--gelu_activation true  \
--enc_attention_dropout 0.1  \
--n_heads 8 \
--dropout '0.1'  \
--master_port 43133 --reload_model $MLM_MODEL --word_pred 0.3 --debug_train True
