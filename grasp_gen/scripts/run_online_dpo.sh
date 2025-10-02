#!/bin/bash

# 在线DPO训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 配置参数
CONFIG_PATH="./configs"
CONFIG_NAME="default"
TASK_CONFIG="task/grasp_gen_ur_online.yaml"

# 创建输出目录
OUTPUT_DIR="/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/online_dpo"
mkdir -p $OUTPUT_DIR

# 设置实验名称
EXP_NAME="online_dpo_$(date +%Y%m%d_%H%M%S)"

# 运行在线DPO训练
python online_dpo_trainer.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    task=$TASK_CONFIG \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    gpu=0 \
    lora_rank=16 \
    beta=1.0 \
    online.buffer_size=10000 \
    online.min_batch_size=32 \
    online.update_frequency=10 \
    online.data_collection_interval=0.1 \
    online.save_interval=100 \
    dpo.online_learning_rate=1e-5 \
    dpo.reference_model_update_freq=1000 \
    data_collection.enable_human_feedback=true \
    data_collection.enable_simulation=true \
    data_collection.enable_real_robot=false \
    monitoring.enable_tensorboard=true \
    buffer_management.type="priority" \
    update_strategy.type="adaptive" \
    hydra.run.dir=$OUTPUT_DIR/$EXP_NAME

echo "在线DPO训练已启动，实验名称: $EXP_NAME"
echo "输出目录: $OUTPUT_DIR/$EXP_NAME"
echo "TensorBoard日志: $OUTPUT_DIR/$EXP_NAME/tb_logs"
echo "模型检查点: $OUTPUT_DIR/$EXP_NAME/ckpts" 