#!/bin/bash


mpirun --allow-run-as-root --hostfile ./hostfile -np 16 -npernode 8 -mca btl_tcp_if_include eth0 \
	-x NCCL_DEBUG=WARN \
	bash -x ./examples

