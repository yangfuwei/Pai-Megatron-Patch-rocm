export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATETIME=`date +'%y-%m-%d_%H-%M-%S'`

PROJ_DIR=/home/amd/fuweiy/Pai-Megatron-Patch-rocm

TP=4
PP=1
EP=2
WS=$((EP * TP * PP))
NUM_EXP=8

# BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=8
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=256
# BATCH_SIZE=16 # 32
# GLOBAL_BATCH_SIZE=256

# SEQ_LEN=128
# PAD_LEN=128
SEQ_LEN=2048
PAD_LEN=2048

TYPE=bf16

MODEL_SIZE=7B
CVT_CKPT=/home/amd/fuweiy/cache/models/Mixtral-8x7B-v0.1 
# MODEL_SIZE=22B
# CVT_CKPT=mistralai/Mixtral-8x22B-v0.1 

DATASET=/home/amd/fuweiy/cache/datasets/wudao_mistralbpe_content_document 
OUTPUT_DIR=/home/amd/fuweiy/cache/models/megatron/Megatron-Mixtral-8x7B-tp${TP}-pp${PP}-ep${EP}
LOG=/${PWD}/../output_mcore_mistral/log/mixtral_8x${MODEL_SIZE}-tp${TP}-pp${PP}-ep${EP}-exp${NUM_EXP}-seq${SEQ_LEN}-b${BATCH_SIZE}-${TYPE}_${DATETIME}.txt

# # convert model weight
ORI_CKPT=/home/amd/fuweiy/cache/models/Mixtral-8x7B-v0.1 
CVT_CKPT=/home/amd/fuweiy/cache/models/Mixtral-8x7B-v0.1-to-mcore-tp${TP}-pp${PP}-ep${EP}-exp${NUM_EXP}-ws${WS}
cd toolkits/model_checkpoints_convertor/mistral
sh hf2mcore_convertor.sh \
8x7B \
$ORI_CKPT \
../../../     \
$ORI_CKPT \
$CVT_CKPT \
${TP}  \
${PP}  \
0  \
${NUM_EXP}  \
2  \
${EP} \
false \
${WS}

cd $PROJ_DIR
cd examples/mistral

sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
${MODEL_SIZE}   \
${BATCH_SIZE}    \
${GLOBAL_BATCH_SIZE} \
1e-5   \
1e-6   \
${SEQ_LEN}  \
${PAD_LEN}  \
0   \
${TYPE}  \
${TP}   \
${PP}  \
sel  \
true   \
false  \
true   \
false   \
true \
100000  \
${DATASET} \
$CVT_CKPT \
100000000   \
10000   \
${OUTPUT_DIR} 2>&1 | tee $LOG
