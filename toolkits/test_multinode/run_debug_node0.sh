MASTER_ADDR=10.11.8.152
MASTER_PORT=10086
WORLD_SIZE=8

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT debug_gloo.py 
