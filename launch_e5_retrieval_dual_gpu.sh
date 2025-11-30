#!/bin/bash
#
# Launch E5 Dense Retrieval Server with Dual GPU acceleration
#
# This script starts a FastAPI server at http://127.0.0.1:8000/retrieve
# The index will be sharded across 2 GPUs to avoid OOM
# Make sure to run this BEFORE starting RL training
#

# Set GPUs (使用 GPU 0 和 1 for retrieval, 保留其他 GPU 给训练)
# 你可以根据实际需要修改为其他 GPU 组合，如 "2,3" 或 "6,7"
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
# 降低临时内存限制以避免多worker并发时OOM
# 40GB时在并发请求下仍会崩溃，降到30GB确保稳定
export FAISS_GPU_TEMP_MEM_GB=30

echo "============================================================================"
echo "Starting E5 Dense Retrieval Server with Dual GPU"
echo "============================================================================"
echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Model: $RETRIEVER_MODEL"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Top-K: $TOPK"
echo "============================================================================"
echo "NOTE: Index will be sharded across available GPUs"
echo "      Each GPU will handle ~30GB of the 60GB index"
echo "============================================================================"

# Launch retrieval server with Dual GPU acceleration
python search_r1/search/retrieval_server.py \
    --index_path $INDEX_FILE \
    --corpus_path $CORPUS_FILE \
    --topk $TOPK \
    --retriever_name $RETRIEVER_NAME \
    --retriever_model $RETRIEVER_MODEL \
    --faiss_gpu

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"