EXP_NAME=$1

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env train_ddm.py \
#                 hydra/job_logging=none hydra/hydra_logging=none \
#                 exp_name=${EXP_NAME} \
#                 diffuser=ddpm \
#                 diffuser.loss_type=l1 \
#                 diffuser.steps=100 \
#                 model=unet_grasp \
#                 task=grasp_gen_ur \
#                 task.dataset.normalize_x=true \
#                 task.dataset.normalize_x_trans=true
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --use_env train_ddm.py \
                hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true