sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
32 \
0   \
0  \
2048  \
2048  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
true   \
false   \
true \
1000  \
../../datasets/wudao_mistralbpe_content_document \
../../model_utils/tokenizer_7b \
100000000   \
20000   \
/home/amd/fuweiy/cache/checkpoints/mixtral-8x7b-tp4pp1ep1-2nodes-fromscratch \
0 \
10.11.8.152 \
2 >& 1 | tee MI300-Mixtral8x7B-fromscratch-single.log

# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep4-exp8-ws64
# mistralai/Mixtral-8x22B-v0.1 \
# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep1-exp8-ws16-debug
