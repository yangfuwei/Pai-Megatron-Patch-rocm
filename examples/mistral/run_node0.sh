TP=8
PP=2
EP=1
MBS=1
GBS=32

sh run_finetune_mcore_mistral_withGA.sh  \
dlc  \
../../ \
22B   \  # model size, 7B or 22B
$MBS    \    
$GBS \
1e-5   \   #lr
1e-6   \   #min lr
2048  \
2048  \
0   \
bf16  \
${TP}   \
${PP}  \
sel  \
true   \
false  \
true   \
false   \
true \
1500  \
../../datasets/alpaca_zh-mistral-train.json \
../../datasets/alpaca_zh-mistral-valid.json \
../../model_utils/tokenizer_7b \
100000000   \
20000   \
./checkpoints/mixtral-8x22b-tp8pp2ep1-2nodes-fromscratch \
0 \
10.11.8.152 \
${EP} \
2 >& 1 | tee MI300-Mixtral8x22B-tp${TP}pp${PP}ep${EP}-mbs${MBS}-gbs${GPS}.log

# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep4-exp8-ws64
# mistralai/Mixtral-8x22B-v0.1 \
# /home/amd/fuweiy/cache/models/megatron/Mixtral-8x22B-v0.1-to-mcore-tp8-pp2-ep1-exp8-ws16-debug
