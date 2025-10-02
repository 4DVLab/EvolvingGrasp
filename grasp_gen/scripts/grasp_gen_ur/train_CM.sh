CKPT=$1
EXP_NAME=$2
export CUDA_VISIBLE_DEVICES=0
python train_CM.py hydra/job_logging=none hydra/hydra_logging=none \
                load_ckpt_dir=${CKPT}    \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true