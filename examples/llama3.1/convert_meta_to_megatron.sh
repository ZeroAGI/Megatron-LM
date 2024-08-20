#!/bin/bash

LLAMA_META_FORMAT_DIR="/mnt2/model/Meta-Llama-3.1-70B/original"
MEGATRON_FORMAT_DIR="/mnt2/model/megatron/pp2_pad_bf16/Meta-Llama-3.1-70B"
TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-70B/original/tokenizer.model"

#LLAMA_META_FORMAT_DIR="/mnt2/model/Meta-Llama-3.1-8B/original"
#MEGATRON_FORMAT_DIR="/mnt2/model/megatron/Meta-Llama-3.1-8B"
#TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-8B/original/tokenizer.model"

TP=8
PP=2

python tools/checkpoint/convert.py \
	   --model-type GPT \
	   --loader llama_mistral \
	   --saver mcore \
	   --checkpoint-type meta \
	   --model-size llama3-70B \
	   --load-dir $LLAMA_META_FORMAT_DIR \
	   --save-dir ${MEGATRON_FORMAT_DIR} \
	   --tokenizer-model ${TOKENIZER_MODEL} \
	   --target-tensor-parallel-size ${TP} \
	   --target-pipeline-parallel-size ${PP} \
	   --make-vocab-size-divisible-by 49825 \
	   --bf16
           #--true-vocab-size 398596
	   #--make-vocab-size-divisible-by 49825 
	   #--bf16
