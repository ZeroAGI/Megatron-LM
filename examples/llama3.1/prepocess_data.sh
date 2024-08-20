#!/bin/bash

DATA_PATH="/mnt2/dataset/wiki_zh_2019/wiki_zh/total.jsonl"
TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-70B/original/tokenizer.model"

ARGS=(
--input ${DATA_PATH}
--output-prefix my-llama
--tokenizer-type Llama3Tokenizer
--tokenizer-model ${TOKENIZER_MODEL}
--workers 32
--append-eod
)

python tools/preprocess_data.py \
	${ARGS[@]}
