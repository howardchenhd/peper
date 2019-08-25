# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel

from src.fp16 import network_to_half


def load_binarized(path):
    return torch.load(path)['dico']


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")
    parser.add_argument("--batch_size", type=int, default=70, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    # dico for source language / target language
    parser.add_argument("--src_data", type=str, default="", help="Source language")
    parser.add_argument("--tgt_data", type=str, default="", help="Source language")

    # beam size
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--input_path", type=str, default="")

    # parser.add_argument("--norm_emb", type=bool_flag, default=False,
    #             help="if do layer_norm on embedding")
    # parser.add_argument("--enc_langemb", type=bool_flag, default=False,
    #     help="if add lang_emb")
    # parser.add_argument("--dec_langemb", type=bool_flag, default=False,
    #     help="if add lang_emb")
    #
    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    model_params.add_pred = ""
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    src_dico = load_binarized(params.src_data)
    tgt_dico = load_binarized(params.tgt_data)

    encoder = TransformerModel(model_params, src_dico, is_encoder=True, with_output=False).cuda().eval()
    decoder = TransformerModel(model_params, tgt_dico, is_encoder=False, with_output=True).cuda().eval()
    
    if all([k.startswith('module.') for k in reloaded['encoder'].keys()]):
        reloaded['encoder'] = {k[len('module.'):]: v for k, v in reloaded['encoder'].items()}
        reloaded['decoder'] = {k[len('module.'):]: v for k, v in reloaded['decoder'].items()}

    encoder.load_state_dict(reloaded['encoder'],strict=False)
    decoder.load_state_dict(reloaded['decoder'],strict=False)


    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]
    
    # # float16

    # # read sentences from stdin
    src_sent = []
    input_f = open(params.input_path, 'r')
    for line in input_f:
        line = line.strip()
        assert len(line.strip().split()) > 0
        src_sent.append(line)
    logger.info("Read %i sentences from stdin. Translating ..." % len(src_sent))

    f = io.open(params.output_path, 'w', encoding='utf-8')

    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = [torch.LongTensor([src_dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)

        # encode source batch and translate it
        encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        encoded = [enc.transpose(0, 1)  for enc in encoded]
        decoded, dec_lengths = decoder.generate(encoded, lengths.cuda(), params.tgt_id, max_len=int(1.5 * lengths.max().item() + 10))


        # convert sentences to words
        for j in range(decoded.size(1)):

            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = src_sent[i + j].strip()
            target = " ".join([tgt_dico[sent[k].item()] for k in range(len(sent))])
            #sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source, target))
            f.write(target + "\n")

    f.close()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    assert params.output_path #and not os.path.isfile(params.output_path)
    assert os.path.isfile(params.input_path), params.input_path
    # translate
    with torch.no_grad():
        main(params)