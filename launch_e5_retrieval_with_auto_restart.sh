#!/bin/bash
#
# Launch E5 Dense Retrieval Server with Auto-Restart
#
# 当检索器因内存泄漏崩溃时，自动重启
# 训练脚本会在检索器重启期间等待（最多5分钟）
#

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Paths
SAVE_PATH=data/wiki-corpus
INDEX_FILE=$SAVE_PATH/e5_Flat.index
CORPUS_FILE=$SAVE_PATH/wiki-18.jsonl

# E5 model
RETRIEVER_NAME=e5
RETRIEVER_MODEL=intfloat/e5-base-v2

# Top-k documents to retrieve
TOPK=3

# FAISS GPU temporary memory (GB per GPU)
export FAISS_GPU_TEMP_MEM_GB=30

# Log directory
LOG_DIR=logs
mkdir -p $LOG_DIR

echo "============================================================================"
echo "Starting E5 Dense Retrieval Server with Auto-Restart"
echo "============================================================================"
echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Model: $RETRIEVER_MODEL"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "FAISS Temp Memory: ${FAISS_GPU_TEMP_MEM_GB}GB per GPU"
echo "Top-K: $TOPK"
echo "============================================================================"
echo "Auto-restart enabled: Server will restart automatically if it crashes"
echo "============================================================================"

RESTART_COUNT=0

while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/retrieval_${TIMESTAMP}.log"

    echo ""
    echo "[$(date)] Starting retrieval server (restart #${RESTART_COUNT})..."
    echo "[$(date)] Log file: ${LOG_FILE}"

    # Launch retrieval server
    python search_r1/search/retrieval_server.py \
        --index_path $INDEX_FILE \
        --corpus_path $CORPUS_FILE \
        --topk $TOPK \
        --retriever_name $RETRIEVER_NAME \
        --retriever_model $RETRIEVER_MODEL \
        --faiss_gpu \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?

    echo ""
    echo "[$(date)] Retrieval server exited with code: ${EXIT_CODE}"

    # Check if this is a graceful shutdown (Ctrl+C)
    if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ]; then
        echo "[$(date)] Graceful shutdown detected. Exiting auto-restart loop."
        break
    fi

    echo "[$(date)] Server crashed! Waiting 5 seconds before restart..."
    sleep 5

    echo "[$(date)] Restarting retrieval server..."
done

echo "============================================================================"
echo "Auto-restart loop ended"
echo "Total restarts: ${RESTART_COUNT}"
echo "============================================================================"