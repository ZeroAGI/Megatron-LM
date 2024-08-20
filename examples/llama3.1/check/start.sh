#!/bin/bash

rm -rf /mnt2/project/rank-worker/project
mkdir -p /mnt2/project/rank-worker/project
cp -r /mnt_share/workspace/project/llama3/ /mnt2/project/rank-worker/project

cd /mnt2/project/rank-worker/project/llama3/ && pip3 install .

cp -r /mnt_share/workspace/project/Megatron-LM/ /mnt2/project/rank-worker/project

cd /mnt2/project/rank-worker/project/Megatron-LM/ && pip3 install .
