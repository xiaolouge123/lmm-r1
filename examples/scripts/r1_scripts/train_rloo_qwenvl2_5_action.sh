set -x

export CUDA_VISIBLE_DEVICES="3,4,5,6,7"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

RUN_NAME="Qwen2.5-VL-3B-Inst-Action-RLOO-r1-zero"
PATH_TO_MODEL="/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-3B-Instruct"
PATH_TO_DATASET="/data/true_nas/zfs_share1/zyc/workspace/lmm-r1/examples/data/AMEX_acton_rl_chatml.json"
OUTPUT_DIR="/data/true_nas/zfs_share1/zyc/expr"


cleanup() {
    echo "执行清理操作，杀死子进程..."
    kill "$childpid" 2>/dev/null  # 忽略 "No such process" 错误
    wait "$childpid" 2>/dev/null # Wait for the process to terminate, ignoring errors
    echo "子进程已终止。"
}

# 使用 trap 命令在 EXIT 和 ERR 信号时调用 cleanup 函数
trap cleanup EXIT ERR


if [ ! -d "${OUTPUT_DIR}/${RUN_NAME}" ]; then
    mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"
fi

python -m openrlhf.models.remote_rm.action_verifier --dataset $PATH_TO_DATASET --input_key prompt --prompt-template chatml > "${OUTPUT_DIR}/${RUN_NAME}/remote_rm.log" 2>&1 &

childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --temp-dir ~/.cache/ray

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/true_nas/zfs_share1/zyc/projects/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PATH_TO_MODEL \
   --save_path $OUTPUT_DIR/$RUN_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 512 \
   --temperature 1 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $PATH_TO_DATASET \
   --input_key prompt \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path $OUTPUT_DIR/$RUN_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $OUTPUT_DIR/$RUN_NAME/logs

ray stop