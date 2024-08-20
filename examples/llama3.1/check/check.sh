#!/bin/bash

# python3 -c "import torch; print(torch.cuda.is_available())"

/usr/local/mpi/bin/mpirun --allow-run-as-root -hostfile ../hostfile -np 18 -npernode 1 -mca btl_tcp_if_include eth0 python3 -c "import torch; import socket; print(f'{socket.gethostname()}: {torch.cuda.is_available()}')"
