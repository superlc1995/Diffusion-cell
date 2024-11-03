LOG_DIR="--log_dir [save folder] --ann_dir [3*3 layout maps folder] --copula_dir [GMM maps folder] --point_single [1*1 layout maps folder]"
TRAIN_FLAGS="--batch_size 5 --save_interval 10000 --lr 2e-5 --p2_gamma 1 --p2_k 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=2 python image_train_BRCA_gmm.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS