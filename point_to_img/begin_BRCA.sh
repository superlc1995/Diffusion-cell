LOG_DIR="--log_dir /data/superlc/diff_cell/models/BRCA/pt_to_img_ker_brca --resume_checkpoint /home/chenli/cell_detection/guided-diffusion_snr/64_256_upsampler.pt"
TRAIN_FLAGS="--batch_size 3 --save_interval 10000 --lr 2e-5 --p2_gamma 1 --p2_k 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --large_size 512 --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=6 python super_res_train_brca.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS