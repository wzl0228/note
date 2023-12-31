PRE_SEQ_LEN=256
LR=2e-2

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --do_train \
    --train_file data0805/Ques/train.json \
    --validation_file data0805/Ques/dev.json \
    --prompt_column text \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-Ques-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 1320 \
    --max_target_length 1320 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 600 \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

