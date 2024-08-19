# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vssm'
# Model name
_C.MODEL.NAME = 'vssm_base_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = '.../vssmbase_dp06_ckpt_epoch_241.pth'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.5
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# MMpretrain models for test
_C.MODEL.MMCKPT = False

# VSSM parameters
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.DEPTHS = [ 2, 2, 27, 2 ]
_C.MODEL.VSSM.EMBED_DIM = 128
_C.MODEL.VSSM.SSM_D_STATE = 16
_C.MODEL.VSSM.SSM_RATIO = 2.0
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
_C.MODEL.VSSM.MLP_RATIO = 0.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.DOWNSAMPLE = "v1"
_C.MODEL.VSSM.PATCHEMBED = "v1"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args=None):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args !=None:
        update_config(config, args)
    else:
        config = config
    return config
