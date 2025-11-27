#!/bin/bash
#
# Launch E5 Dense Retrieval Server with GPU acceleration
#
# This script starts a FastAPI server at http://127.0.0.1:8000/retrieve
# Make sure to run this BEFORE starting RL training
#

# Set GPU (使用GPU 4，预留5,6,7给训练)
export CUDA_VISIBLE_DEVICES=4

# Paths
SAVE_PATH=data/wiki-corpus
INDEX_FILE=$SAVE_PATH/e5_Flat.index
CORPUS_FILE=$SAVE_PATH/wiki-18.jsonl

# E5 model
RETRIEVER_NAME=e5
RETRIEVER_MODEL=intfloat/e5-base-v2

# Top-k documents to retrieve
TOPK=3

echo "============================================================================"
echo "Starting E5 Dense Retrieval Server"
echo "============================================================================"
echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Model: $RETRIEVER_MODEL"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Top-K: $TOPK"
echo "============================================================================"

# Launch retrieval server with GPU acceleration
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