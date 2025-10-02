# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --use_env train_ddm.py \
#                 hydra/job_logging=none hydra/hydra_logging=none \
#                 exp_name=${EXP_NAME} \
#                 diffuser=ddpm \
#                 diffuser.loss_type=l1 \
#                 diffuser.steps=100 \
#                 model=unet_grasp \
#                 task=grasp_gen_ur \
#                 task.dataset.normalize_x=true \
#                 task.dataset.normalize_x_trans=true
#!/bin/bash

EXP_NAME=$1
MAIN_GPU=${2:-0}
# 设置端口范围
START_PORT=29501
END_PORT=29510

# 查找可用端口
find_free_port() {
    for port in $(seq $START_PORT $END_PORT); do
        if ! lsof -i :$port > /dev/null; then
            echo $port
            return 0
        fi
    done
    echo "No free port found in range $START_PORT-$END_PORT"
    exit 1
}

# 获取一个可用端口
MASTER_PORT=$(find_free_port)

# 设置环境变量
export MASTER_PORT
# 根据主卡设置CUDA_VISIBLE_DEVICES，确保只使用4张卡
# export CUDA_VISIBLE_DEVICES=0
# '''
if [ "$MAIN_GPU" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ "$MAIN_GPU" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1,0,2,3
elif [ "$MAIN_GPU" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2,0,1,3
elif [ "$MAIN_GPU" -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=3,0,1,2
else
    echo "Invalid MAIN_GPU selection. Choose between 0 and 3."
    exit 1
fi
# '''
# 检查并修复 torchrun 的 shebang 行
TORCHRUN_PATH="/inspurfs/group/mayuexin/zym/miniconda3/envs/3d/bin/torchrun"
PYTHON_PATH="/inspurfs/group/mayuexin/zym/miniconda3/envs/3d/bin/python"

if ! grep -q "^#!${PYTHON_PATH}" "${TORCHRUN_PATH}"; then
    sed -i "1s|.*|#!${PYTHON_PATH}|" "${TORCHRUN_PATH}"
    echo "Fixed shebang line in ${TORCHRUN_PATH}"
fi

# 打印调试信息
echo "Using port: $MASTER_PORT"
echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# 启动分布式训练
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=localhost:$MASTER_PORT train_ddm.py \
    hydra/job_logging=none hydra/hydra_logging=none \
    exp_name=${EXP_NAME} \
    diffuser=ddpm \
    diffuser.loss_type=l1 \
    diffuser.steps=100 \
    model=unet_grasp \
    task=grasp_gen_ur \
    task.dataset.normalize_x=true \
    task.dataset.normalize_x_trans=true
