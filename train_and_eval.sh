#!/bin/bash

# Check if arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <is_UD> <task>"
    exit 1
fi

# Get the current date and time
current_date_time=$(date '+%Y-%m-%d_%H-%M-%S')

# Create a new directory named "experiments" concatenated with the current date and time
dir_name="experiment_${current_date_time}"
mkdir ${dir_name}

# Run the training
training_args="--nnodes 1 --nproc_per_node 2 --rdzv-backend c10d --rdzv-endpoint localhost:0 llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /dataslow/toti/llama-2/Llama-2-7b-chat-hf --pure_bf16 --dataset $1 --num_epochs 1 --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --output_dir ${dir_name}"
torchrun ${training_args} > ${dir_name}/training_output.txt

# Run the evaluation
eval_args="--eval_dir ${dir_name} --model_name /dataslow/toti/llama-2/Llama-2-7b-chat-hf --peft_model_name ${dir_name} --max_new_tokens 500 --max_sentences_per_doc 10 --top_p 0.9 --top_k 30 --do_sample True --temperature 0.25 --is_UD $2 --task $3"
python coref_eval.py ${eval_args} > ${dir_name}/eval_output.txt

# Store the arguments
echo "Training arguments: ${training_args}" > ${dir_name}/params.txt
echo "Eval arguments: ${eval_args}" >> ${dir_name}/params.txt