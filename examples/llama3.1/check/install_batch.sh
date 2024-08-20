#!/bin/bash

/usr/local/mpi/bin/mpirun --allow-run-as-root -hostfile ../hostfile -np 18 -npernode 1 -mca btl_tcp_if_include eth0 bash -x start.sh

