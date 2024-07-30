import os
import torch
import torch.distributed as dist

dist.init_process_group(backend='gloo')

rank = os.getenv('RANK', -1)

print(rank, 'entering barrier...')
torch.distributed.barrier()
print(rank, 'through barrier...')
