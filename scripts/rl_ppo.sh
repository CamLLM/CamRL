nproc_per_node=4

model_path=""
reward_model_path=""
dataset_path="AI-ModelScope/alpaca-gpt4-data-zh#20000"
output_dir=""

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type ppo \
    --model $model_path \
    --reward_model $reward_model_path \
    --train_type full \
    --dataset $dataset_path \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --response_length 512 \
    --temperature 0.7 \
    --dataset_num_proc 4 \
    --save_only_model true \
    --deepspeed zero2 # 这里换成zero3也可以
