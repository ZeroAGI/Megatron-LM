#!/bin/bash

LLAMA_META_FORMAT_DIR="/mnt2/model/Meta-Llama-3.1-70B/original"
MEGATRON_FORMAT_DIR="/mnt2/model/megatron/pp2/Meta-Llama-3.1-70B"
TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-70B/original/tokenizer.model"

LLAMA_META_FORMAT_DIR="/mnt2/model/Meta-Llama-3.1-8B/original"
MEGATRON_FORMAT_DIR="/mnt2/model/megatron/Meta-Llama-3.1-8B"
TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-8B/original/tokenizer.model"

TP=8
PP=1

python tools/checkpoint/convert.py \
	   --model-type GPT \
	   --loader llama2 \
	   --saver megatron \
	   --checkpoint-type meta \
	   --model-size 8B \
	   --load-dir $LLAMA_META_FORMAT_DIR \
	   --save-dir ${MEGATRON_FORMAT_DIR} \
	   --tokenizer-model ${TOKENIZER_MODEL} \
	   --target-tensor-parallel-size 8 \
	   --target-pipeline-parallel-size 1
	   
	   #--bf16
