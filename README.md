# Diffusion-cell

The code for MICCAI2024 paper: [Spatial diffusion for cell layout generation](https://arxiv.org/pdf/2409.03106)

## 1. Training and infering for cell layout generation ##

Training layout generation framework with GMM density maps on BRCA dataset:

### BRCA Dataset preprocess ###
```
python gen_gmm_map.py # generating GMM density maps for BRCA dataset
mkdir point_single
python single_point_gen.py # generating 1*1 layout map for BRCA dataset
mkdir point_den
python point_den_gen.py # generating 3*3 layout map for BRCA dataset
```

### Training cell layout generation framework ###

```
LOG_DIR="--log_dir [save folder] --ann_dir [3*3 layout maps folder] --copula_dir [GMM maps folder] --point_single [1*1 layout maps folder]"
TRAIN_FLAGS="--batch_size 5 --save_interval 10000 --lr 2e-5 --p2_gamma 1 --p2_k 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=2 python image_train_BRCA_gmm.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

Here we have:

```
[GMM maps folder] = gmm_den/
[3*3 layout maps folder] = point_den/
[1*1 layout maps folder] = point_single/
```

Or use:

```
bash begin_BRCA_gmm.sh
```

### Generating cell layouts with trained model ###

```
python gen_BRCA_gmm.py --model_path [pretrained model] --save_folder [dir for saving generated layouts]
```
<!-- To train layout generation framework with GMM density maps on BRCA dataset,  -->


## 2. Training and infering for layout conditioned pathology image generation ##


The layout-to-image training and generating diffusion model is in folder ```point_to_img```.

To start the training process of layout-to-image, just ```bash begin_BRCA.sh``` in folder ```point_to_img```.

The inference of layout-to-image model is in ```point_to_img/gen_BRCA_gmm.py```.


## 3. To be updated ##

1. Pretrained models
2. Spatial FID
<!-- 2.  -->
<!-- The  -->