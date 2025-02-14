set -x

export CUDA_VISIBLE_DEVICES="3,4,5,6,7"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

RUN_NAME="Qwen2.5-VL-3B-Inst-Action-RLOO-r1-zero"
PATH_TO_MODEL="/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-7B-Instruct"
PATH_TO_DATASET="/data/true_nas/zfs_share1/zyc/expr/data/amex_train_split.jsonl"
OUTPUT_DIR="/data/true_nas/zfs_share1/zyc/expr"


if [ ! -d "${OUTPUT_DIR}/${RUN_NAME}" ]; then
    mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"
fi

python -m openrlhf.models.remote_rm.action_verifier --dataset $PATH_TO_DATASET --input_key prompt --prompt-template chatml > "${OUTPUT_DIR}/${RUN_NAME}/remote_rm.log" 2>&1 &

childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --temp-dir ~/.cache/ray

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/root/projects/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.7 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PATH_TO_MODEL \
   --save_path $OUTPUT_DIR/$RUN_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
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