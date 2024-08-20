#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.1.0.6 # localhost
MASTER_PORT=6000
LOCAL_ADDR=10.1.0.5
NUM_NODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_documen

TP=8
TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-70B/original/tokenizer.model"
CHECKPOINT_DIR="/mnt2/model/megatron/Meta-Llama-3.1-70B"
CHECKPOINT_DIR="/mnt2/model/megatron/pp2/Meta-Llama-3.1-70B"
SAVE_DIR="/mnt2/model/megatron_save/Meta-Llama-3.1-70B"

TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-8B/original/tokenizer.model"
CHECKPOINT_DIR="/mnt2/model/megatron/Meta-Llama-3.1-8B"
CHECKPOINT_DIR="/mnt2/model/megatron/pp2/Meta-Llama-3.1-8B"
SAVE_DIR="/mnt2/model/megatron_save/Meta-Llama-3.1-8B"


TENSORBOARD_LOGS_PATH="./tensorboard"
DATA_PATH="$(pwd)/my-llama_text_document" # "/mnt2/dataset/wiki_zh_2019/wiki_zh/AA"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

ARGS=(
    --distributed-backend nccl \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 2 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load $CHECKPOINT_DIR \
    --save $SAVE_DIR \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --tensorboard-dir $TENSORBOARD_LOGS_PATH \
    --distributed-timeout-minutes 2 \
    --save-interval 100 
    #--no-bias-dropout-fusion
)

DATA_ARGS=(
--data-path $DATA_PATH
)

TRAINING_ARGS=(
	    --micro-batch-size 1
	        --global-batch-size 1
		        --train-iters 100
			    --weight-decay 0.1
			        --adam-beta1 0.9
				    --adam-beta2 0.95
				        --init-method-std 0.006
					    --clip-grad 1.0
						    --lr 6.0e-5
						        --lr-decay-style cosine
							    --min-lr 6.0e-6
							        --lr-warmup-fraction .001
								    --lr-decay-iters 430000
							    )

# EVAL_AND_LOGGING_ARGS=(
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --save $SAVE_DIR \
#     --load $CHECKPOINT_PATH \
#     --eval-iters 10 \
#     --tensorboard-dir $TENSORBOARD_LOGS_PATH 
# )

/home/tulingtuxian/miniconda3/envs/megatron/bin/torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]}
