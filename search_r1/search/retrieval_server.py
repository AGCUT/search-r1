import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

def load_corpus(corpus_path: str):
    """加载语料库到内存字典，加速文档访问"""
    corpus_dict = {}
    print(f"Loading corpus from {corpus_path}...")
    import time
    start_time = time.time()

    with open(corpus_path, 'r') as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            corpus_dict[idx] = doc

            # 每100万条显示进度
            if (idx + 1) % 1000000 == 0:
                print(f"  Loaded {idx + 1:,} documents...")

    elapsed = time.time() - start_time
    print(f"Loaded {len(corpus_dict):,} documents in {elapsed:.1f} seconds")
    print(f"Memory usage: ~{len(corpus_dict) * 1000 / (1024**2):.0f} MB estimated")
    return corpus_dict

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs, docid_to_idx=None):
    """
    加载文档，过滤无效索引

    Args:
        corpus: 语料库字典 (int -> doc)
        doc_idxs: 文档索引列表（可能是整数或字符串）
        docid_to_idx: 可选的 docid -> 行号映射字典（用于 BM25）

    Returns:
        results: 有效文档列表
        valid_indices: 有效文档在原始列表中的位置（用于同步过滤得分）
    """
    results = []
    valid_indices = []

    for i, idx in enumerate(doc_idxs):
        doc = None

        # 尝试直接转换为整数（FAISS 的情况）
        try:
            idx_int = int(idx)
            if idx_int >= 0:  # FAISS 在找不到邻居时返回 -1
                doc = corpus.get(idx_int)
        except (ValueError, TypeError):
            # 不是整数，尝试通过 docid_to_idx 映射（BM25 的情况）
            if docid_to_idx is not None and idx in docid_to_idx:
                idx_int = docid_to_idx[idx]
                doc = corpus.get(idx_int)

        # 只添加有效文档
        if doc is not None:
            results.append(doc)
            valid_indices.append(i)

    return results, valid_indices

def load_model(model_path: str, use_fp16: bool = False, use_cuda: bool = True):
    """加载模型，支持 CPU 和 GPU"""
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # 检测 CUDA 可用性
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        if use_fp16:
            model = model.half()
    else:
        if use_cuda:
            print("Warning: CUDA not available, using CPU for model")
        # CPU 模式下不使用 fp16（CPU 不支持 half）

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.use_cuda = torch.cuda.is_available()

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, use_cuda=self.use_cuda)
        self.model.eval()
        self.device = next(self.model.parameters()).device  # 获取模型所在设备

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True, encode_batch_size=4) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # 分批编码以避免 GPU 显存不足
        all_embeddings = []
        for i in range(0, len(query_list), encode_batch_size):
            batch_queries = query_list[i:i + encode_batch_size]

            inputs = self.tokenizer(batch_queries,
                                    max_length=self.max_length,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    )
            # 将输入移到模型所在的设备（CPU 或 GPU）
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if "T5" in type(self.model).__name__:
                # T5-based retrieval model
                decoder_input_ids = torch.zeros(
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                ).to(inputs['input_ids'].device)
                output = self.model(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                query_emb = output.last_hidden_state[:, 0, :]
            else:
                output = self.model(**inputs, return_dict=True)
                query_emb = pooling(output.pooler_output,
                                    output.last_hidden_state,
                                    inputs['attention_mask'],
                                    self.pooling_method)
                if "dpr" not in self.model_name.lower():
                    query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

            query_emb = query_emb.detach().cpu().numpy()
            all_embeddings.append(query_emb)

            del inputs, output, query_emb
            # 只在 CUDA 可用时清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 合并所有批次的结果
        final_emb = np.concatenate(all_embeddings, axis=0)
        final_emb = final_emb.astype(np.float32, order="C")

        return final_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
            # 构建 docid -> 行号的映射
            # 假设 corpus 的 id 字段对应 Pyserini 的 docid
            self.docid_to_idx = {}
            print("Building docid -> index mapping for BM25...")
            for idx, doc in self.corpus.items():
                if 'id' in doc:
                    self.docid_to_idx[doc['id']] = idx
                # 同时支持直接用字符串化的行号作为 docid
                self.docid_to_idx[str(idx)] = idx
            print(f"Built mapping for {len(self.docid_to_idx)} docids")
        else:
            self.corpus = None
            self.docid_to_idx = None
        self.max_process_num = 8

    def _check_contain_doc(self):
        """检查索引是否包含原始文档内容"""
        try:
            # 先做一次搜索获取真实的 docid
            test_hits = self.searcher.search("test", 1)
            if len(test_hits) == 0:
                # 索引为空或无法搜索，假设不含文档
                return False
            # 使用搜索返回的真实 docid
            test_docid = test_hits[0].docid
            doc = self.searcher.doc(test_docid)
            return doc is not None and doc.raw() is not None
        except:
            return False

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []

        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            # 索引包含原文，直接从索引读取
            results = []
            valid_scores = []
            for hit, score in zip(hits, scores):
                try:
                    raw = self.searcher.doc(hit.docid).raw()
                    if raw is not None:
                        content = json.loads(raw)['contents']
                        results.append({
                            'title': content.split("\n")[0].strip("\""),
                            'text': "\n".join(content.split("\n")[1:]),
                            'contents': content
                        })
                        valid_scores.append(score)
                except:
                    # 跳过无法解析的文档
                    continue
        else:
            # 索引不包含原文，从外部语料加载
            docids = [hit.docid for hit in hits]
            results, valid_indices = load_docs(self.corpus, docids, self.docid_to_idx)
            # 同步过滤得分
            valid_scores = [scores[i] for i in valid_indices]

        if return_score:
            return results, valid_scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            # 创建多 GPU 资源配置
            ngpus = faiss.get_num_gpus()
            print(f"Found {ngpus} GPUs for faiss")

            if ngpus == 0:
                print("Warning: faiss_gpu=True but no GPU available, falling back to CPU")
            elif ngpus > 0:
                # 为每个 GPU 创建资源对象，限制临时显存
                gpu_resources = []

                # 动态计算临时显存大小：假设每张 GPU 有 80GB，预留 20GB 给其他用途
                # 用户可以通过环境变量 FAISS_GPU_TEMP_MEM_GB 自定义
                temp_mem_gb = int(os.environ.get('FAISS_GPU_TEMP_MEM_GB', '55'))
                temp_mem_bytes = temp_mem_gb * 1024 * 1024 * 1024
                print(f"Setting temp memory per GPU: {temp_mem_gb} GB")

                for i in range(ngpus):
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(temp_mem_bytes)
                    gpu_resources.append(res)

                # 将索引加载到 GPU
                if ngpus == 1:
                    print(f"Loading index to single GPU...")
                    # 单 GPU 使用 GpuClonerOptions（faiss.index_cpu_to_gpu 只接受该类型）
                    co_single = faiss.GpuClonerOptions()
                    co_single.useFloat16 = True
                    self.index = faiss.index_cpu_to_gpu(gpu_resources[0], 0, self.index, co_single)
                else:
                    print(f"Sharding index to {ngpus} GPUs...")
                    # 多 GPU 使用 GpuMultipleClonerOptions
                    co_multi = faiss.GpuMultipleClonerOptions()
                    co_multi.useFloat16 = True
                    co_multi.shard = True
                    self.index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, self.index, co_multi)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        # load_docs 返回 (results, valid_indices)
        results, valid_indices = load_docs(self.corpus, idxs)
        # 同步过滤得分
        valid_scores = [scores[i] for i in valid_indices]
        if return_score:
            return results, valid_scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []

        # 为了避免 FAISS GPU 显存不足，每次只编码和搜索少量查询
        # 多worker并发时可能同时有100+查询，必须设为1确保稳定
        encode_batch = 1  # E5 编码批次大小（设为1最保守，避免并发OOM）
        search_batch = 1  # FAISS 搜索批次大小（每次只搜索1个）

        for start_idx in tqdm(range(0, len(query_list), encode_batch), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + encode_batch]
            # 分批编码
            batch_emb = self.encoder.encode(query_batch, encode_batch_size=encode_batch)

            # 对每个查询单独搜索，避免 FAISS GEMM 显存爆炸
            for i in range(len(query_batch)):
                single_emb = batch_emb[i:i+1]  # (1, 768)
                query_scores, query_idxs = self.index.search(single_emb, k=num)
                query_scores = query_scores[0].tolist()
                query_idxs = query_idxs[0].tolist()

                # load_docs 返回 (results, valid_indices)
                query_results, valid_indices = load_docs(self.corpus, query_idxs)
                # 同步过滤得分
                query_valid_scores = [query_scores[j] for j in valid_indices]
                results.append(query_results)
                scores.append(query_valid_scores)

                del single_emb, query_scores, query_idxs
                # 只在 CUDA 可用时清空缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del batch_emb, query_batch

        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# 全局变量，在 startup 时初始化
config = None
retriever = None

# 并发控制：同时只处理1个检索请求，避免GPU显存OOM
retrieval_semaphore = asyncio.Semaphore(1)

@app.on_event("startup")
async def startup_event():
    """FastAPI startup 事件：初始化配置和检索器"""
    global config, retriever

    # 如果已经通过 __main__ 初始化，则跳过
    if config is not None and retriever is not None:
        return

    # 从环境变量读取配置（支持 uvicorn 直接启动）
    import os
    config = Config(
        retrieval_method=os.environ.get('RETRIEVAL_METHOD', 'e5'),
        index_path=os.environ.get('INDEX_PATH', './data/wiki-corpus/e5_Flat.index'),
        corpus_path=os.environ.get('CORPUS_PATH', './data/wiki-corpus/wiki-18.jsonl'),
        retrieval_topk=int(os.environ.get('RETRIEVAL_TOPK', '3')),
        faiss_gpu=os.environ.get('FAISS_GPU', 'false').lower() == 'true',
        retrieval_model_path=os.environ.get('RETRIEVAL_MODEL', 'intfloat/e5-base-v2'),
        retrieval_pooling_method='mean',
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=32,
    )

    print("Initializing retriever...")
    retriever = get_retriever(config)
    print("Retriever initialized successfully")

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    使用Semaphore限制并发，同时只处理1个请求，避免多worker并发导致GPU OOM

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    # 获取信号量，确保同时只有1个请求在处理
    async with retrieval_semaphore:
        if not request.topk:
            request.topk = config.retrieval_topk  # fallback to default

        # Perform batch retrieval
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=True
        )

        # Format response
        resp = []
        for i, single_result in enumerate(results):
            if request.return_scores:
                # If scores are returned, combine them with results
                combined = []
                for doc, score in zip(single_result, scores[i]):
                    combined.append({"document": doc, "score": score})
                resp.append(combined)
            else:
                resp.append(single_result)

        return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=32,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
