sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
32 \
1e-5   \
1e-6  \
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
false # grouped gemm

