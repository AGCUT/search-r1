#!/bin/bash
#
# Launch BM25 Sparse Retrieval Server (CPU only)
#
# BM25 is faster and doesn't require GPU, but may be less accurate than E5
#

# No GPU needed for BM25
export CUDA_VISIBLE_DEVICES=""

# Paths
SAVE_PATH=data/wiki-corpus
INDEX_FILE=$SAVE_PATH/bm25
CORPUS_FILE=$SAVE_PATH/wiki-18.jsonl

# BM25 retriever
RETRIEVER_NAME=bm25

# Top-k documents to retrieve
TOPK=3

echo "============================================================================"
echo "Starting BM25 Sparse Retrieval Server"
echo "============================================================================"
echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Top-K: $TOPK"
echo "Mode: CPU only (no GPU required)"
echo "============================================================================"

# Launch BM25 retrieval server (CPU only)
python search_r1/search/retrieval_server.py \
    --index_path $INDEX_FILE \
    --corpus_path $CORPUS_FILE \
    --topk $TOPK \
    --retriever_name $RETRIEVER_NAME

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"