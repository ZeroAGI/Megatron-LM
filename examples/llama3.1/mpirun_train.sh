#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

HOSTFLIE="/mnt_share/workspace/hostfile_bak"

GPUS_PER_NODE=8
# Change for multinode config
#MASTER_ADDR=10.1.0.11 # localhost
MASTER_ADDR=$(ifconfig eth0 | grep 'inet ' | awk '{print $2}')
MASTER_PORT=6000
#LOCAL_ADDR=10.1.0.5
NUM_NODES=$(wc -l $HOSTFLIE | awk '{print $1}')
#NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

TP=8
PP=2

MBS=14
GBS=$(($MBS*$NUM_NODES*GPUS_PER_NODE/TP/PP))


# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH= #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_documen

TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-70B/original/tokenizer.model"
CHECKPOINT_DIR="/mnt2/model/megatron/Meta-Llama-3.1-70B"
CHECKPOINT_DIR="/mnt2/model/megatron/pp2/Meta-Llama-3.1-70B"
SAVE_DIR="/mnt2/model/megatron_save/Meta-Llama-3.1-70B"

TOKENIZER_MODEL="/mnt_share/workspace/model/Meta-Llama-3.1-70B/original/tokenizer.model"
CHECKPOINT_DIR="/mnt_share/workspace/model/megatron/pp2_pad_bf16/Meta-Llama-3.1-70B/"
#CHECKPOINT_DIR="/mnt_share/workspace/model/megatron/pp4_pad_bf16/Meta-Llama-3.1-70B/"
SAVE_DIR="/mnt2/project/rank-worker/output/Meta-Llama-3.1-70B"

#TOKENIZER_MODEL="/mnt2/model/Meta-Llama-3.1-8B/original/tokenizer.model"
#CHECKPOINT_DIR="/mnt2/model/megatron/pp2/Meta-Llama-3.1-8B"
#SAVE_DIR="/mnt2/model/megatron_save/Meta-Llama-3.1-8B"

TENSORBOARD_LOGS_PATH="./tensorboard"
DATA_PATH="/mnt2/project/new/Megatron-LM/my-llama_text_document" # "/mnt2/dataset/wiki_zh_2019/wiki_zh/AA"
DATA_PATH="/mnt_share/workspace/data/my-llama_text_document"
DATA_PATH="/mnt2/project/rank-worker/project/Megatron-LM/my-llama_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

ARGS=(
    --use-mcore-models \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
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
    --distributed-timeout-minutes 60 \
    --save-interval 500 \
    --log-throughput \
    --log-interval 1 \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
    --use-distributed-optimizer \
    --bf16 \
    --tp-comm-overlap \
    --overlap-grad-reduce \
    --overlap-param-gather
    #--recompute-method uniform \
    #--recompute-num-layers 1 \
    #--recompute-granularity full \
    #--num-workers 8
    #--tp-comm-overlap
    #--recompute-activations
    #--no-bias-dropout-fusion
)

DATA_ARGS=(
--data-path $DATA_PATH
)

TRAINING_ARGS=(
	    --micro-batch-size ${MBS}
	        --global-batch-size ${GBS}
		        --train-iters 10000
			  --eval-iters 0
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

#torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
#    ${ARGS[@]} \
#    ${DATA_ARGS[@]} \
#    ${TRAINING_ARGS[@]}


mpirun --allow-run-as-root -hostfile ${HOSTFLIE} -np ${WORLD_SIZE} -npernode ${GPUS_PER_NODE} -mca btl_tcp_if_include eth0 \
        -x CUDA_HOME="$CUDA_HOME" \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NCCL_DEBUG=WARN \
	-x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
	-x NCCL_P2P_DISABLE=0 \
	-x NCCL_IB_DISABLE=0 \
	-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	python pretrain_gpt.py \
	${ARGS[@]} \
	${DATA_ARGS[@]} \
	${TRAINING_ARGS[@]}


