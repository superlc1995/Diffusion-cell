import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import stopit
from pathlib import Path
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util_brca import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        # model_path="",
        sample_dir="",
        # model_path="/data07/shared/chenli/gen_diff/part_A/models/pt_to_pt_shha/model070000.pt"
        model_path=""
        save_folder = ''
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args([])
args.num_channels = 256
args.num_res_blocks=2
# args.num_heads=4
args.num_head_channels=64
args.attention_resolutions='32,16,8'
args.class_cond=True
args.use_scale_shift_norm=True
args.resblock_updown=True
args.use_fp16=False
args.learn_sigma=True
args.diffusion_steps=1000
args.noise_schedule='linear'
args.p2_gamma = 1
args.p2_k = 1
args.image_size =256
args.timestep_respacing = 'ddim100'
args.use_ddim =True
# dist_util.setup_dist()
# logger.configure()


# Specify the directory path
dir_path = Path(args.save_folder )

# Check if the directory exists
if not dir_path.exists():
    # Create the directory if it doesn't exist
    dir_path.mkdir(parents=True)
    print(f"Directory '{dir_path}' created.")
else:
    print(f"Directory '{dir_path}' already exists.")
# logger.log("creating model...")
# model, diffusion = sr_create_model_and_diffusion(
#     **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
# )
# model.load_state_dict(
#     dist_util.load_state_dict(args.model_path, map_location="cpu")
# )
# model.to(dist_util.dev())
# if args.use_fp16:
#     model.convert_to_fp16()
# model.eval()

dist_util.setup_dist()
logger.configure()

logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()

sample_fn = (
    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
)

def class_to_count(inpt):
    sppl = np.array([73.0, 148.0, 194.0, 276.0])
    if inpt < sppl[0]:
        return 0
    elif (inpt < sppl[1]) and (inpt >= sppl[0]):
        return 1
    elif (inpt < sppl[2]) and (inpt >= sppl[1]):
        return 2
    elif (inpt < sppl[3]) and (inpt >= sppl[2]):
        return 3
    else:
        return 4


def denoise_fun(inp):
    all_labels = np.zeros([256,256,3])
    unique_list = []
    for ik in range(3):
        image = ( inp[0][ik] > 0.4 ).numpy().astype(int)
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        u_val = np.unique(labels)
        for ikii in u_val:
            if np.sum(labels == ikii) <= 5:
                labels[labels == ikii] = 0
        all_labels[:,:,ik] = labels
        unique_list.append( len(np.unique( labels)) )

    return all_labels, np.sum(unique_list)

res_list = []

for elemm in range(0,5):
    valid_num = 0
    while valid_num <200:

        classes = th.randint(
            low=0, high=10, size=(1,), device=dist_util.dev()
        )
        classes[0] =elemm


        model_kwargs = {}
        model_kwargs["y"] = classes
        sample = sample_fn(
            model,
            (1, 6, 256,256),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # out_res = np.zeros([256,256,3])
        # indicat = 1
        # for ijj in range(3):
        with stopit.ThreadingTimeout(5) as context_manager:
            labels_res, num_cell = denoise_fun(sample.detach().cpu() )
            # print(num_cell)
            # print(elemm)





        if (context_manager.state == context_manager.EXECUTED):




            np.save( args.save_folder + '/pt_' + str(valid_num) + '_' + str(elemm)  + '.npy', labels_res)
            # th.save(  sample[0][3:6].detach().cpu(), '/data07/shared/chenli/gen_diff/BRCA/gen_den_gmm/den_' + str(valid_num) + '_' + str(elemm)  + '.pth' )
            valid_num = valid_num + 1
    
        # Did code timeout?
        elif context_manager.state == context_manager.TIMED_OUT:
            print("DID NOT FINISH...")

