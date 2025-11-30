#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构造微调训练集脚本
将预处理的向量数据转换为模型微调所需的训练数据格式

支持问答题格式（无选项）

使用方法: python build_finetune_dataset.py --base-dir /path/to/train
"""

import argparse
import numpy as np
import pandas as pd
import re
import json
import os
from tqdm import trange


def load_data(base_dir, data_prefix='train_b'):
    """加载预处理的数据"""
    print(f"加载数据...")

    # 加载问题数据
    question_file = os.path.join(base_dir, 'train.jsonl')
    df_question = pd.read_json(question_file, lines=True)
    print(f"  - 问题数量: {len(df_question)}")

    # 检查数据格式（选择题还是问答题）
    has_options = 'options' in df_question.columns
    print(f"  - 数据类型: {'选择题' if has_options else '问答题'}")

    # 加载问题向量
    question_vector_file = f'all_{data_prefix}_question_vectors.npy'
    question_vector = np.load(question_vector_file)
    print(f"  - 问题向量: {question_vector.shape}")

    # 加载图片向量和映射
    img_vector_file = f'{data_prefix}_pdf_img_vectors.npy'
    img_mapping_file = f'{data_prefix}_pdf_img_page_num_mapping.csv'
    pdf_image_vectors = np.load(img_vector_file)
    pdf_image_page_num_mapping = pd.read_csv(img_mapping_file)
    print(f"  - 图片向量: {pdf_image_vectors.shape}")

    return df_question, question_vector, pdf_image_vectors, pdf_image_page_num_mapping, has_options


def get_similar_image_embedding(df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                                 question_vector, question_idx, top_k):
    """获取与问题最相似的图片页码"""
    document_name = df_question.document[question_idx].split('.')[0]
    vec_idx = pdf_image_page_num_mapping[pdf_image_page_num_mapping['file_name'] == document_name]['index'].values

    if len(vec_idx) == 0:
        return []

    candidate_vec = pdf_image_vectors[vec_idx]
    query_vec = question_vector[question_idx]

    # 计算余弦相似度
    cos_sim = np.dot(candidate_vec, query_vec) / (
        np.linalg.norm(candidate_vec, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    # 获取最相似的top_k个索引
    top_k = min(top_k, len(cos_sim))
    top_k_indices = np.argsort(cos_sim)[-top_k:][::-1]
    retrived_idx = vec_idx[top_k_indices]
    retrived_page_num = pdf_image_page_num_mapping.loc[retrived_idx]['page_num'].to_list()

    return retrived_page_num


def build_qa_prompt(question, options=None):
    """
    构建问答prompt

    Args:
        question: 问题文本
        options: 选项列表（可选，问答题时为None）

    Returns:
        prompt前缀和后缀
    """
    if options:
        # 选择题格式
        prompt_prefix = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的单选题。\n"
        prompt_prefix += f"【我的问题】【{question}】\n"
        prompt_prefix += f"【选项】【{' '.join(options)}】\n"
        prompt_prefix += "专利内容为：\n"
        prompt_suffix = "\n\n请你分析专利内容后，回答我的单选题，直接回答正确选项字母，你的答案为：\n"
    else:
        # 问答题格式
        prompt_prefix = "你是一个专利内容分析专家，请根据我提供的专利内容回答问题。\n"
        prompt_prefix += f"【问题】{question}\n"
        prompt_prefix += "【专利内容】\n"
        prompt_suffix = "\n\n请根据以上专利内容，简洁准确地回答问题：\n"

    return prompt_prefix, prompt_suffix


def get_image_answer(base_dir, df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                     question_vector, document_name, question, question_idx, options=None, top_k=2):
    """构造图像问答的训练样本"""

    prompt_prefix, prompt_suffix = build_qa_prompt(question, options)

    retrived_page_list = get_similar_image_embedding(
        df_question, pdf_image_page_num_mapping, pdf_image_vectors,
        question_vector, question_idx, top_k
    )

    if len(retrived_page_list) == 0:
        return None, []

    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for page_num in retrived_page_num:
        image_file = os.path.join(base_dir, 'pdf_img', document_name.split('.')[0], f'{page_num}.jpg')
        if os.path.exists(image_file):
            retrived_list.append(image_file)

    if len(retrived_list) == 0:
        return None, []

    query = prompt_prefix
    for _ in retrived_list:
        query += '<image>'
    query += prompt_suffix

    return query, retrived_list


def get_mix_answer_img(base_dir, df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                       question_vector, document_name, pic_page_num, question, question_idx,
                       options=None, top_k=2):
    """构造混合图像问答的训练样本（问题指向特定页面+召回相似页面）"""

    if options:
        # 选择题格式
        question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答我的问题。\n"
        question1 += f"【我的问题】【{question}】\n"
        question1 += f"【选项】【{' '.join(options)}】\n"
        question1 += "该问题直接指向的专利页内容为：\n"
        question3 = "\n\n请你分析专利内容后，回答我的单选题，直接回答正确选项字母，你的答案为：\n"
    else:
        # 问答题格式
        question1 = "你是一个专利内容分析专家，请根据我提供的专利内容回答问题。\n"
        question1 += f"【问题】{question}\n"
        question1 += "【该问题指向的专利页内容】\n"
        question3 = "\n\n请根据以上专利内容，简洁准确地回答问题：\n"

    retrived_page_list = get_similar_image_embedding(
        df_question, pdf_image_page_num_mapping, pdf_image_vectors,
        question_vector, question_idx, top_k
    )

    # 排序
    retrived_page_num = sorted(retrived_page_list)
    retrived_list = []
    for page_num in retrived_page_num:
        image_file = os.path.join(base_dir, 'pdf_img', document_name.split('.')[0], f'{page_num}.jpg')
        if os.path.exists(image_file):
            retrived_list.append(image_file)

    question2 = "\n【其他相关专利内容】\n"

    # 问题指向的图片
    main_image = os.path.join(base_dir, 'pdf_img', document_name.split('.')[0], f'{pic_page_num}.jpg')

    if not os.path.exists(main_image):
        return None, []

    query = question1 + '<image>'
    images = [main_image]

    if len(retrived_list) > 0:
        query += question2
        for img in retrived_list:
            if img != main_image:  # 避免重复
                query += '<image>'
                images.append(img)

    query += question3

    return query, images


def write_to_jsonl(result_dict, output_file):
    """写入jsonl文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')


def build_dataset(base_dir, output_dir='.', data_prefix='train_b'):
    """构建训练数据集"""

    # 加载数据
    df_question, question_vector, pdf_image_vectors, pdf_image_page_num_mapping, has_options = load_data(
        base_dir, data_prefix
    )

    # 输出文件
    img_train_file = os.path.join(output_dir, 'train_dataset_for_image.jsonl')

    # 清空已有文件
    if os.path.exists(img_train_file):
        os.remove(img_train_file)

    print(f"\n开始构建训练集...")
    print(f"  - 输出文件: {img_train_file}")

    success_count = 0
    skip_count = 0

    for i in trange(len(df_question), desc="构建训练样本"):
        question = df_question.loc[i, 'question']
        document_name = df_question.loc[i, 'document']
        true_answer = df_question.loc[i, 'answer']

        # 获取选项（如果有的话）
        options = df_question.loc[i, 'options'] if has_options else None

        # 判断问题类型
        if "第" in question and "页" in question:
            # 问题指向特定页面
            page_match = re.findall(r"第(\d+)页", question)
            if page_match:
                pic_page_num = int(page_match[0])
                query, images = get_mix_answer_img(
                    base_dir, df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                    question_vector, document_name, pic_page_num, question, i, options
                )
            else:
                query, images = get_image_answer(
                    base_dir, df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                    question_vector, document_name, question, i, options
                )
        else:
            # 普通问题，使用图像召回
            query, images = get_image_answer(
                base_dir, df_question, pdf_image_page_num_mapping, pdf_image_vectors,
                question_vector, document_name, question, i, options
            )

        if query is None or len(images) == 0:
            skip_count += 1
            continue

        result_dict = {
            'question_idx': i,
            'document': document_name,
            'query': query,
            'response': true_answer,
            'images': images
        }
        write_to_jsonl(result_dict, img_train_file)
        success_count += 1

    print(f"\n构建完成!")
    print(f"  - 成功: {success_count} 条")
    print(f"  - 跳过: {skip_count} 条")
    print(f"  - 输出: {img_train_file}")


def main():
    parser = argparse.ArgumentParser(description='构造微调训练集')
    parser.add_argument('--base-dir', type=str,
                        default='/usr/yuque/guo/pdf_processer/patent_b/train',
                        help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录')
    parser.add_argument('--data-prefix', type=str, default='train_b',
                        help='数据文件前缀')

    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"{'='*60}")
    print(f"构造微调训练集")
    print(f"{'='*60}")
    print(f"训练数据目录: {args.base_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据文件前缀: {args.data_prefix}")

    build_dataset(args.base_dir, args.output_dir, args.data_prefix)


if __name__ == '__main__':
    main()