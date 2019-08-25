# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel, Bridge  # , TRANSFORMER_LAYER_PARAMS
import pdb

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            pass
            # s = params.reload_model.split(',')
            # assert len(s) == 2
            # assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))
            if 'encoder' in reloaded:
                reloaded = reloaded['encoder']
            else:
                reloaded = reloaded['model']

            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded,strict=False)

        if params.fix_enc_layers != -1:
            assert params.fix_enc_layers >= 0
            if params.fix_enc_layers == params.enc_layers:
                params.fix_enc = True


            model.position_embeddings.weight.requires_grad = False
            model.lang_embeddings.weight.requires_grad = False
            model.layer_norm_emb.weight.requires_grad = False
            model.layer_norm_emb.bias.requires_grad = False
            model.embeddings.weight.requires_grad = False

            for layer in range(params.fix_enc_layers):
                for name, p in model.named_parameters():
                    if  '.{}.'.format(layer) in name:
                        p.requires_grad = False

        logger.debug("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico['src'], is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico['tgt'], is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico['src'], word2id, embeddings)
            set_pretrain_emb(decoder, dico['tgt'], word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}

                enc_reload = { k: v for k,v in enc_reload.items() if k in encoder.state_dict() }

                for k, v in encoder.state_dict().items():
                    if k not in enc_reload:
                        logger.warning("Reassignment parameters:{}".format(k))
                        enc_reload[k] = v
                encoder.load_state_dict(enc_reload,strict=True)
            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.dec_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]

                dec_reload = { k: v for k,v in dec_reload.items() if k in decoder.state_dict() }

                for k, v in decoder.state_dict().items():
                    if k not in dec_reload:
                        logger.warning("Reassignment parameters:{}".format(k))
                        dec_reload[k] = v

                decoder.load_state_dict(dec_reload)

        encoder.dico = dico['src']
        decoder.dico = dico['tgt']
        encoder.pred_layer = None


        if params.fix_enc_layers != -1:
            assert params.fix_enc_layers >= 0
            if params.fix_enc_layers == params.enc_layers:
                params.fix_enc = True

            encoder.position_embeddings.weight.requires_grad = False
            encoder.lang_embeddings.weight.requires_grad = False
            encoder.layer_norm_emb.weight.requires_grad = False
            encoder.layer_norm_emb.bias.requires_grad = False
            encoder.embeddings.weight.requires_grad = False

            for layer in range(params.fix_enc_layers):
                for name, p in encoder.named_parameters():
                    if  '.{}.'.format(layer) in name:
                        p.requires_grad = False

        if params.fix_enc:
            for p in encoder.parameters():
                p.requires_grad = False


        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))
        return encoder.cuda(), decoder.cuda()


def build_bridge(params):
    bridge = Bridge(params).cuda()
    logger.info("Number of parameters (bridge): %i" % sum([p.numel() for p in bridge.parameters() if p.requires_grad]))
    return bridge