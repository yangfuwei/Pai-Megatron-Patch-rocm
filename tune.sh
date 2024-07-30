# export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# export HIP_VISIBLE_DEVICES=0,1,2,3,4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATETIME=`date +'%y-%m-%d_%H-%M-%S'`

PROJ_DIR=/workspace/Pai-Megatron-Patch-rocm

TP=4
PP=1
EP=1
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
CVT_CKPT=mistralai/Mixtral-8x7B-v0.1 
# MODEL_SIZE=22B
# CVT_CKPT=mistralai/Mixtral-8x22B-v0.1 

DATASET=/workspace/mixtral-datasets/wudao/wudao_mistralbpe_content_document 
OUTPUT_DIR=/workspace/output_mcore_mistral
LOG=/${PWD}/../output_mcore_mistral/log/mixtral_8x${MODEL_SIZE}-tp${TP}-pp${PP}-ep${EP}-exp${NUM_EXP}-seq${SEQ_LEN}-b${BATCH_SIZE}-${TYPE}_${DATETIME}.txt


cd $PROJ_DIR/examples/mistral

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

TRAIN_ITERS=6
TRAIN_TOKENS=$(( ${TRAIN_ITERS} * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} ))
WARMUP_TOKENS=0
nohup ROCBLAS_LAYER=4 bash run_pretrain_mcore_mistral.sh  \
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
${TRAIN_TOKENS}   \
${WARMUP_TOKENS}   \
${OUTPUT_DIR} > log_nohup.log 2>&1 & # grep "\- { rocblas_function:" | uniq | tee rocblas.yaml

rm -rf ${OUTPUT_DIR}/checkpoint

# python ../../pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py rocblas.yaml --cuda_device 0 1 2 3 4 5 6 7

# export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv
# export PYTORCH_TUNABLEOP_TUNING=0
# export PYTORCH_TUNABLEOP_ENABLED=1
# TRAIN_TOKENS=100000000
# WARMUP_TOKENS=10000
# bash run_pretrain_mcore_mistral.sh  \
# dsw  \
# ../../ \
# ${MODEL_SIZE}   \
# ${BATCH_SIZE}    \
# ${GLOBAL_BATCH_SIZE} \
# 1e-5   \
# 1e-6   \
# ${SEQ_LEN}  \
# ${PAD_LEN}  \
# 0   \
# ${TYPE}  \
# ${TP}   \
# ${PP}  \
# sel  \
# true   \
# false  \
# true   \
# false   \
# true \
# 100000  \
# ${DATASET} \
# $CVT_CKPT \
# ${TRAIN_TOKENS}   \
# ${WARMUP_TOKENS}   \
# ${OUTPUT_DIR} 2>&1 | tee $LOG
