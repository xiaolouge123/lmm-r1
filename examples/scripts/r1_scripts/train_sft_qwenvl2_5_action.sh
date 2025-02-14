set -x

RUN_NAME="Qwen2.5-VL-3B-Inst-Action-Long-COT-Cold-Start"
PATH_TO_MODEL="/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-7B-Instruct"
PATH_TO_DATASET="/data/true_nas/zfs_share1/zyc/workspace/simpleRL-reason/prompt_dataset.jsonl"
OUTPUT_DIR="/data/true_nas/zfs_share1/zyc/expr"

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_${RUN_NAME}.txt"
export CUDA_VISIBLE_DEVICES="3,4,5,6,7"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi


