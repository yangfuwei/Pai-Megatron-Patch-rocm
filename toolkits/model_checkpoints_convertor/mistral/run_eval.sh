#accelerate launch --multi-gpu \
#    --num_processes 2 \
#    -m lm_eval \
lm_eval \
    --model hf \
    --model_args pretrained=/home/amd/fuweiy/cache/models/Mixtral-8x22B-v0.1,parallelize=True \
    --tasks hellaswag,winogrande,arc_challenge,mmlu \
    --batch_size 4
