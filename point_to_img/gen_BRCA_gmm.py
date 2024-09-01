import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import skimage.io
import torch.nn.functional as F
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util_brca import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        model_path="/data/superlc/diff_cell/models/BRCA/pt_to_img_ker_brca/model500000.pt", # The position of pretrained model
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args([])
args.num_channels = 192
args.num_res_blocks=2
args.num_heads=4
args.num_head_channels=64
args.attention_resolutions='32,16,8'
args.class_cond=False
args.use_scale_shift_norm=True
args.resblock_updown=True
args.use_fp16=True
args.learn_sigma=True
args.diffusion_steps=1000
args.noise_schedule='linear'
args.p2_gamma = 1
args.p2_k = 1
args.large_size =512
args.timestep_respacing = 'ddim250'
args.use_ddim =True

dist_util.setup_dist()
logger.configure()

logger.log("creating model...")
model, diffusion = sr_create_model_and_diffusion(
    **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()



for elemm in range(5):

    for num_la in range(200):
        img1 = th.load('/data/superlc/diff_cell/BRCA/refined_generation/gen_pt_gmm/pt_' + str(num_la) + '_' + str(elemm) + '.pt')[0] # load in layout, I have an example in this folder (pt_20_2.pt)
        img1 = th.tensor(img1).permute(2,0,1)[None,:].float()
        # img1 = th.load('/data/superlc/diff_cell/PSU/kde_norm/den/pt_' + str(num_la) + '_' + str(elemm) + '.pth')
        # img1 = th.tensor(img1)
        # img1 = img1[None,None,:].float()

        img1 = img1 *2 - 1
        img1 = F.pad(img1, [24,24,24,24], "constant", -1)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        model_kwargs = {}
        model_kwargs['low_res'] = img1
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = sample_fn(
            model,
            (1, 3,512,512),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        skimage.io.imsave('/data/superlc/diff_cell/BRCA/gen_img/gmm_norm/img_' + str(num_la) + '_' + str(elemm) + '.png', ((sample[0].detach().cpu().permute(1,2,0).numpy()  + 1)/2 * 255).astype(np.uint8)) # save the generated image