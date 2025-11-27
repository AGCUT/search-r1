# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the implementation for the CCKS 2025 PDF QA Multimodal Competition (TianChi competition), focusing on patent document question answering using multimodal (text + image) approaches.

Competition URL: https://tianchi.aliyun.com/competition/entrance/532357/information
Full explanation: https://zhuanlan.zhihu.com/p/1937478905532506972

## Repository Structure

- **round_a/**: Initial round implementation (less organized, exploratory)
- **round_b/**: Final round implementation (production pipeline)
- **choice_pipeline/**: Alternative pipeline approach with choice-based RAG
- **pic/**: Documentation images

## Core Architecture

### Multimodal RAG Pipeline

The system uses a multimodal Retrieval-Augmented Generation (RAG) approach:

1. **Document Processing**: PDFs are converted to images (600 DPI) and OCR text is extracted
2. **Embedding Generation**: Both images and text are embedded using GME-Qwen2-VL-7B model (3584-dimensional vectors)
3. **Vector Retrieval**: Cosine similarity search retrieves relevant pages for each question
4. **Answer Generation**: Fine-tuned Qwen2.5-VL-32B model generates answers based on retrieved multimodal content
5. **Answer Refinement**: A secondary pass extracts concise answers and applies style consistency

### Key Components

**GmeQwen2VL (gme_inference.py)**
- Multimodal embedding model wrapper around Qwen2-VL-7B
- Generates 3584-dim embeddings for both text and images
- Handles image resizing with smart_resize to maintain aspect ratio within token limits
- Used for both training and test set vectorization

**Vector Storage**
- Images: `{train/test}_b_pdf_img_vectors.npy` (numpy arrays)
- Questions: `all_{train/test}_b_question_vectors.npy`
- Mappings: CSV files linking vector indices to page numbers and file names

**Inference Pipeline (test_b_style_refer_215.py)**
- Loads fine-tuned Qwen2.5-VL-32B model with LoRA
- Uses vLLM for efficient inference with multi-GPU support
- Retrieves top-2 similar pages using cosine similarity
- Classifies questions to determine if additional context is needed
- Applies style consistency using similar training examples

## Round B Workflow (Production Pipeline)

### 1. Install Dependencies
```bash
pip install ms-swift
# Download models using ModelScope (魔搭)
```

### 2. Preprocessing (b_train_test_preprocess.py)
This is the most time-consuming step. It processes both training and test sets:

**For Training Set:**
```bash
python b_train_test_preprocess.py
```
- Converts PDFs to JPG images (600 DPI) from `/data/coding/patent_b/train/documents/`
- Generates image embeddings using GME-Qwen2-VL and saves to `train_b_pdf_img_vectors.npy`
- Creates page number mappings in `train_b_pdf_img_page_num_mapping.csv`
- Generates question embeddings and saves to `all_train_b_question_vectors.npy`

**For Test Set:**
- Same process but saves with `test_` prefix
- Results: `test_b_pdf_img_vectors.npy`, `test_b_pdf_img_page_num_mapping.csv`, `all_test_b_question_vectors.npy`

### 3. Training Data Construction
```bash
jupyter notebook finetune训练集构造_v2.ipynb
```
Creates the training dataset JSONL file for fine-tuning.

### 4. Model Training (train_vl_32b.sh)
```bash
bash train_vl_32b.sh
```
Fine-tunes Qwen2.5-VL-32B-Instruct with:
- LoRA (rank=8, alpha=32)
- 8 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
- MAX_PIXELS=1229312 (controls image token count ~1280 tokens)
- Learning rate: 1e-4
- 5 epochs, batch size 1 per device, gradient accumulation 16 steps
- Output: `/data/coding/lora_qwen25_vl_32b_b/`

### 5. Generate Results (test_b_style_refer_215.py)
```bash
python test_b_style_refer_215.py
```
Runs inference on test set:
- Loads checkpoint-215 merged model
- Uses 4 GPUs with vLLM (tensor_parallel_size=4)
- MAX_PIXELS=1568000 (~2000 tokens per image)
- Retrieves top-2 similar training questions for style reference
- Classifies questions to determine if only image analysis is needed
- Outputs to `test_b_style_infer_if_need_ck215.jsonl`

## Choice Pipeline (Alternative Approach)

Located in `choice_pipeline/` directory, this is an alternative implementation that treats the task as multiple choice when applicable. Key differences:
- Uses choice-based RAG approach
- Has separate training scripts: `train_vl.sh` and `train_text.sh`
- Test script: `choice_rag3_run_test_set_8145.py`

## Important Implementation Details

### Environment Variables
- `MAX_PIXELS`: Controls image resolution and token count (1229312 for training, 1568000 for inference)
- `CUDA_VISIBLE_DEVICES`: GPU allocation

### Model Paths
Training typically uses paths like:
- Base model: `/data/coding/llm_model/Qwen/Qwen2___5-VL-32B-Instruct`
- LoRA output: `/data/coding/lora_qwen25_vl_32b_b/`
- Merged checkpoint: `/data/coding/lora_qwen25_vl_32b_for_b/v0-20250802-085531/checkpoint-215-merged/`

### Data Paths
- Training data: `/data/coding/patent_b/train/`
- Test data: `/data/coding/patent_b/test/`
- PDFs: `documents/` subdirectory
- Generated images: `pdf_img/` subdirectory

## Round A (Initial Exploration)

The `round_a/` directory contains earlier experimental code:
- Multiple test scripts for OCR and embeddings
- Similar pipeline but less refined
- Includes Jupyter notebooks for data processing and analysis

## Running on Cloud Platforms

When using SSH for long-running processes (like training or preprocessing):
```bash
apt-get update && apt-get install -y tmux
tmux
# Run your command here
```
This prevents SSH disconnections from interrupting execution.

## Data Download

Competition data can be downloaded using ossutil64:
```bash
wget http://gosspublic.alicdn.com/ossutil/1.7.12/ossutil64
chmod 755 ossutil64
./ossutil64 cp oss://[official_data_path] ./
```

## Key Dependencies

- ms-swift: For model training and inference
- vLLM: Fast inference engine for vision-language models
- transformers: HuggingFace model loading
- qwen_vl_utils: Qwen vision utilities
- PyMuPDF (fitz): PDF to image conversion
- numpy, pandas: Data processing
