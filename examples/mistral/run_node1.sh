sh run_pretrain_mcore_mistral.sh  \
dlc  \
../../ \
22B   \
1    \
32 \
1e-5   \
1e-6   \
2048  \
2048  \
0   \
bf16  \
8   \
2  \
sel  \
true   \
false  \
true   \
false   \
true \
1500  \
../../datasets/wudao_mistralbpe_content_document \
../../model_utils/tokenizer_7b \
100000000   \
20000   \
/home/amd/fuweiy/cache/checkpoints/mixtral-8x22b-tp8pp2ep1-2nodes-fromscratch \
1 \
10.11.8.152 \
2 >& 1 | tee MI300-Mixtral8x22B-tp8pp2ep1-fromscratch.log

# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep4-exp8-ws64
# mistralai/Mixtral-8x22B-v0.1 \
# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep1-exp8-ws16-debug
