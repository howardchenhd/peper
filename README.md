# Transformer-Pytorch

The origin code author is [Guillaume Lample](https://github.com/glample).

## Multi-GPU

`
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
`