#!/usr/bin/env bash


python train.py                                \
--exp_name test_deen_mlm                       \
--dump_path ../dumped/                          \
--data_path /data4/bjji/source/github/data/tlm/en_de \
--lgs 'de-en'                                  \
--emb_dim 1024                                 \
--n_layers 6                                   \
--n_heads 8                                    \
--dropout 0.1                                  \
--attention_dropout 0.1                        \
--gelu_activation true                         \
--batch_size 32                                \
--bptt 256                                     \
--optimizer adam,lr=0.0001                     \
--epoch_size 100                            \
--validation_metrics valid_en-de_mlm_acc            \
--mass_steps "de-en,en-de"                     \
--mass_type  "block"                           \
--block_size  0.5 \
--reload_model '/data4/bjji/source/github/model/mlm_base/mlm_ende_1024.pth' \
--stopping_criterion valid_en-de_mlm_acc,20 \
--save_periodic 1 