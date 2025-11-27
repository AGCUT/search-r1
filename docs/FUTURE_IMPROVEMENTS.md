# 未来改进方案

本文档记录了针对当前系统的优化建议和具体实施方法。

## 📊 当前问题分析

### 主要问题
**相对位置关系类问题回答不准确**

**问题表现**:
- 对于"部件A在部件B的哪个方向"这类问题，准确率不够高
- 模型难以准确理解复杂的空间关系

**根本原因**:
1. 整页图像包含过多信息，模型注意力分散
2. 图像分辨率受限（MAX_PIXELS限制）
3. 缺乏显式的推理步骤训练
4. 空间关系推理需要多步骤思考

---

## 🎯 改进方案

## 方案1: 文档结构化 + 精确图片定位

### 📋 目标
- 对于"问某页某图"的问题，只让模型看到精确裁剪的图片
- 提高单图分辨率，使用更高的MAX_PIXELS
- 减少无关信息干扰

### 💡 核心思路
```
原方案: 问题 → 检索整页 → 整页图像(低分辨率) → 模型
新方案: 问题 → 解析图号 → 精确裁剪图片 → 单图(高分辨率) → 模型
```

### 🔧 具体实施步骤

#### 步骤1: 文档结构化解析

**1.1 版面分析**
```python
# 新建文件: ccks2025_pdf_multimodal/round_b/document_structure_parser.py

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2

class DocumentStructureParser:
    """文档结构化解析器"""

    def __init__(self):
        self.figure_detector = self.load_figure_detector()

    def parse_page_structure(self, pdf_path, page_num):
        """
        解析单页的结构，识别图片区域

        Returns:
            {
                'page_num': int,
                'figures': [
                    {
                        'figure_id': str,  # 如 "图1", "图2"
                        'bbox': (x0, y0, x1, y1),  # 边界框
                        'confidence': float
                    }
                ],
                'text_regions': [...],
                'tables': [...]
            }
        """
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        # 方法1: 使用PyMuPDF的图像检测
        figures = self._detect_figures_pymupdf(page)

        # 方法2: 使用OCR识别图号
        figures = self._enhance_with_ocr(page, figures)

        # 方法3: 使用深度学习模型（可选）
        # figures = self._enhance_with_dl_model(page, figures)

        return {
            'page_num': page_num,
            'figures': figures
        }

    def _detect_figures_pymupdf(self, page):
        """使用PyMuPDF检测图片"""
        figures = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            bbox = page.get_image_bbox(img)

            figures.append({
                'figure_id': f'图{img_index + 1}',
                'bbox': bbox,
                'xref': xref,
                'confidence': 1.0
            })

        return figures

    def _enhance_with_ocr(self, page, figures):
        """使用OCR识别图号，提高准确性"""
        from qwen_vl_utils import process_vision_info

        # 转换页面为图像
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 使用OCR识别文本
        ocr_results = self._run_ocr(img)

        # 匹配图号（如"图1"、"Fig.1"）
        import re
        pattern = r'图\s*(\d+)|Fig\.\s*(\d+)'

        for ocr_result in ocr_results:
            text = ocr_result['text']
            match = re.search(pattern, text)
            if match:
                fig_num = match.group(1) or match.group(2)
                # 更新对应图片的信息
                self._update_figure_id(figures, ocr_result['bbox'], f'图{fig_num}')

        return figures

    def extract_figure(self, pdf_path, page_num, figure_id, output_path):
        """
        精确提取某个图片

        Args:
            pdf_path: PDF路径
            page_num: 页码
            figure_id: 图片ID（如"图1"）
            output_path: 输出路径
        """
        structure = self.parse_page_structure(pdf_path, page_num)

        # 查找对应的图片
        target_figure = None
        for fig in structure['figures']:
            if fig['figure_id'] == figure_id:
                target_figure = fig
                break

        if not target_figure:
            raise ValueError(f"未找到 {figure_id}")

        # 提取图片
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        # 裁剪图片区域（添加padding）
        bbox = target_figure['bbox']
        padding = 20  # 添加20像素padding
        clip_rect = fitz.Rect(
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        )

        # 高分辨率渲染
        pix = page.get_pixmap(clip=clip_rect, dpi=600)
        pix.save(output_path)

        return output_path
```

**1.2 批量解析和缓存**
```python
# 新建文件: ccks2025_pdf_multimodal/round_b/batch_structure_parser.py

import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def batch_parse_documents(pdf_dir, output_dir):
    """
    批量解析所有PDF的结构

    Args:
        pdf_dir: PDF目录
        output_dir: 输出目录
    """
    parser = DocumentStructureParser()
    pdf_files = list(Path(pdf_dir).glob('*.pdf'))

    structures = {}

    for pdf_file in tqdm(pdf_files, desc="解析文档结构"):
        doc_name = pdf_file.stem
        doc = fitz.open(str(pdf_file))

        page_structures = []
        for page_num in range(1, doc.page_count + 1):
            structure = parser.parse_page_structure(str(pdf_file), page_num)
            page_structures.append(structure)

        structures[doc_name] = page_structures

        # 保存到JSON
        output_file = Path(output_dir) / f'{doc_name}_structure.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(page_structures, f, ensure_ascii=False, indent=2)

    return structures

# 使用示例
if __name__ == '__main__':
    # 解析训练集
    batch_parse_documents(
        pdf_dir='/data/coding/patent_b/train/documents',
        output_dir='/data/coding/patent_b/train/structures'
    )

    # 解析测试集
    batch_parse_documents(
        pdf_dir='/data/coding/patent_b/test/documents',
        output_dir='/data/coding/patent_b/test/structures'
    )
```

#### 步骤2: 问题解析和图片定位

**2.1 问题解析器**
```python
# 在 test_b_style_refer_215.py 中添加

import re
from typing import Optional, Tuple

def parse_question_figure_reference(question: str) -> Optional[Tuple[int, str]]:
    """
    解析问题中的图片引用

    Args:
        question: 问题文本

    Returns:
        (page_num, figure_id) 或 None

    Examples:
        "观察文件中第6页的图1，编号为12的部件是什么？" → (6, "图1")
        "第3页图2中，部件A在部件B的哪个位置？" → (3, "图2")
    """
    # 模式1: "第X页的图Y"
    pattern1 = r'第\s*(\d+)\s*页.*?图\s*(\d+)'
    match = re.search(pattern1, question)
    if match:
        page_num = int(match.group(1))
        fig_num = match.group(2)
        return (page_num, f'图{fig_num}')

    # 模式2: "第X页" (没有指定图号)
    pattern2 = r'第\s*(\d+)\s*页'
    match = re.search(pattern2, question)
    if match:
        page_num = int(match.group(1))
        return (page_num, None)  # None表示整页

    return None

def should_use_figure_crop(question: str) -> bool:
    """判断是否应该使用图片裁剪"""
    # 关键词：相对位置、方向、空间关系
    position_keywords = ['位置', '方向', '上方', '下方', '左侧', '右侧', '旁边', '之间']
    return any(keyword in question for keyword in position_keywords)
```

**2.2 修改推理脚本**
```python
# 修改 test_b_style_refer_215.py

def get_optimized_image_input(question, document_name, question_idx):
    """
    根据问题智能选择输入图像

    Returns:
        {
            'images': [图像路径列表],
            'max_pixels': 建议的MAX_PIXELS值,
            'mode': 'full_page' 或 'cropped_figure'
        }
    """
    # 解析问题
    fig_ref = parse_question_figure_reference(question)

    if fig_ref and should_use_figure_crop(question):
        page_num, figure_id = fig_ref

        # 加载文档结构
        structure_file = f'/data/coding/patent_b/test/structures/{document_name}_structure.json'
        with open(structure_file, 'r') as f:
            structures = json.load(f)

        page_structure = structures[page_num - 1]

        # 如果指定了图号
        if figure_id:
            # 提取单个图片
            parser = DocumentStructureParser()
            cropped_path = f'/tmp/{document_name}_p{page_num}_{figure_id}.jpg'
            parser.extract_figure(
                pdf_path=f'/data/coding/patent_b/test/documents/{document_name}.pdf',
                page_num=page_num,
                figure_id=figure_id,
                output_path=cropped_path
            )

            return {
                'images': [cropped_path],
                'max_pixels': 2352000,  # 更高分辨率
                'mode': 'cropped_figure'
            }

    # 默认方案：使用整页
    similar_pages = get_similar_image_embedding(document_name, question_idx, top_k=2)
    image_paths = [f'/data/coding/patent_b/test/pdf_img/{document_name}/{p}.jpg'
                   for p in similar_pages]

    return {
        'images': image_paths,
        'max_pixels': 1568000,
        'mode': 'full_page'
    }
```

#### 步骤3: 更新预处理脚本

```bash
# 新建脚本: scripts/01_preprocess_enhanced.sh

#!/bin/bash
# 增强版预处理：包含文档结构化

set -e

echo "=========================================="
echo "增强版数据预处理"
echo "=========================================="

# 步骤1: 原有预处理
bash scripts/01_preprocess.sh

# 步骤2: 文档结构化解析
cd ccks2025_pdf_multimodal/round_b

echo "步骤2: 解析文档结构..."
python batch_structure_parser.py

echo "文档结构解析完成！"
echo "结构文件保存在: /data/coding/patent_b/{train,test}/structures/"
```

### 📊 预期效果

**分辨率提升**:
- 原方案: 整页 1568000 pixels (~2000 tokens)
- 新方案: 单图 2352000+ pixels (~3000+ tokens)
- **提升**: ~50%+

**准确率提升**:
- 预计位置关系问题准确率提升 **10-15%**

### ⚠️ 注意事项

1. **图号识别准确性**: 需要准确识别图号，可能需要多种方法结合
2. **边界框精确性**: 裁剪时需要包含足够的上下文
3. **缓存管理**: 裁剪的图片需要合理缓存
4. **失败回退**: 如果无法裁剪，应该回退到整页方案

---

## 方案2: 数据增强 - 推理链生成

### 📋 目标
- 为训练数据添加推理链（Chain-of-Thought）
- 教会模型一步步推理

### 💡 核心思路
```
原训练数据: 问题 → 答案
新训练数据: 问题 → 推理步骤 → 答案
```

### 🔧 具体实施步骤

#### 步骤1: 使用强模型生成推理链

**1.1 推理链生成器**
```python
# 新建文件: ccks2025_pdf_multimodal/round_b/reasoning_chain_generator.py

from vllm import LLM, SamplingParams
import json

class ReasoningChainGenerator:
    """推理链生成器"""

    def __init__(self, model_path):
        # 使用更强的模型（如Qwen2.5-VL-72B或微调后的模型）
        self.model = LLM(model=model_path, tensor_parallel_size=8)

    def generate_reasoning_chain(self, question, images, ground_truth_answer):
        """
        生成推理链

        Args:
            question: 问题
            images: 图像路径列表
            ground_truth_answer: 正确答案（用于验证）

        Returns:
            {
                'reasoning_steps': [步骤1, 步骤2, ...],
                'final_answer': 答案,
                'confidence': 置信度
            }
        """
        prompt = f"""你是一个专利分析专家。请一步步分析下面的问题，详细说明推理过程。

问题：{question}

请按照以下格式回答：

【分析步骤】
1. 首先，我需要识别图中的关键信息...
2. 然后，我需要确定部件的位置关系...
3. 接下来，我需要...
4. 最后，根据以上分析...

【最终答案】
{ground_truth_answer}

现在开始你的分析："""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image", "image": img} for img in images]
                ]
            }
        ]

        sampling_params = SamplingParams(
            temperature=0.7,  # 稍高的温度获得多样性
            max_tokens=1024,
            top_p=0.9
        )

        response = self.model.generate(messages, sampling_params)
        reasoning_text = response[0].outputs[0].text

        # 解析推理步骤
        steps = self._parse_reasoning_steps(reasoning_text)

        return {
            'reasoning_steps': steps,
            'reasoning_text': reasoning_text,
            'ground_truth': ground_truth_answer
        }

    def _parse_reasoning_steps(self, text):
        """解析推理步骤"""
        import re

        # 提取【分析步骤】部分
        steps_match = re.search(r'【分析步骤】\n(.*?)\n【最终答案】', text, re.DOTALL)
        if not steps_match:
            return []

        steps_text = steps_match.group(1)
        steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', steps_text, re.DOTALL)

        return [step.strip() for step in steps]
```

**1.2 批量生成推理链**
```python
# 新建文件: ccks2025_pdf_multimodal/round_b/batch_generate_reasoning.py

import pandas as pd
from tqdm import tqdm
import json

def batch_generate_reasoning_chains(
    train_data_path,
    output_path,
    sample_rate=0.3  # 对30%的数据生成推理链
):
    """
    批量为训练数据生成推理链

    Args:
        train_data_path: 原始训练数据路径
        output_path: 输出路径
        sample_rate: 采样率（重点采样位置关系类问题）
    """
    generator = ReasoningChainGenerator(
        model_path='/data/coding/llm_model/Qwen/Qwen2___5-VL-32B-Instruct'
    )

    # 加载训练数据
    df = pd.read_json(train_data_path, lines=True)

    # 筛选位置关系类问题（优先生成）
    position_keywords = ['位置', '方向', '上方', '下方', '左侧', '右侧']

    def is_position_question(q):
        return any(kw in q for kw in position_keywords)

    df['is_position'] = df['question'].apply(is_position_question)

    # 采样策略：位置类问题全部生成，其他问题部分生成
    position_samples = df[df['is_position']]
    other_samples = df[~df['is_position']].sample(
        n=int(len(df) * sample_rate),
        random_state=42
    )

    samples_to_process = pd.concat([position_samples, other_samples])

    # 生成推理链
    augmented_data = []

    for idx, row in tqdm(samples_to_process.iterrows(), total=len(samples_to_process)):
        try:
            reasoning = generator.generate_reasoning_chain(
                question=row['question'],
                images=row['images'],
                ground_truth_answer=row['answer']
            )

            # 构造增强后的训练样本
            augmented_sample = {
                'query': row['query'],
                'images': row['images'],
                'response': reasoning['reasoning_text'],  # 包含推理过程的回答
                'original_response': row['answer'],
                'has_reasoning': True
            }

            augmented_data.append(augmented_sample)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # 保存增强后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"生成了 {len(augmented_data)} 个带推理链的样本")
    print(f"保存到: {output_path}")

# 使用示例
if __name__ == '__main__':
    batch_generate_reasoning_chains(
        train_data_path='train_b_dataset_for_image_0801.jsonl',
        output_path='train_b_dataset_with_reasoning.jsonl'
    )
```

#### 步骤2: 混合训练数据

```python
# 新建文件: ccks2025_pdf_multimodal/round_b/merge_training_data.py

def merge_training_data(
    original_data_path,
    reasoning_data_path,
    output_path,
    reasoning_ratio=0.3
):
    """
    混合原始数据和推理链数据

    Args:
        reasoning_ratio: 推理链数据的比例
    """
    # 读取数据
    original_data = []
    with open(original_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))

    reasoning_data = []
    with open(reasoning_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            reasoning_data.append(json.loads(line))

    # 按比例混合
    total_samples = len(original_data)
    reasoning_samples = int(total_samples * reasoning_ratio)

    # 采样
    import random
    random.seed(42)
    selected_reasoning = random.sample(reasoning_data, min(reasoning_samples, len(reasoning_data)))

    # 合并
    merged_data = original_data + selected_reasoning

    # 打乱
    random.shuffle(merged_data)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"合并完成:")
    print(f"  原始样本: {len(original_data)}")
    print(f"  推理链样本: {len(selected_reasoning)}")
    print(f"  总样本: {len(merged_data)}")
    print(f"  保存到: {output_path}")

# 使用示例
if __name__ == '__main__':
    merge_training_data(
        original_data_path='train_b_dataset_for_image_0801.jsonl',
        reasoning_data_path='train_b_dataset_with_reasoning.jsonl',
        output_path='train_b_dataset_merged.jsonl'
    )
```

### 📊 预期效果

**数据增强**:
- 原训练集: 1000样本
- 增强后: 1300样本 (100%原始 + 30%推理链)

**准确率提升**:
- 预计整体准确率提升 **5-8%**
- 位置关系问题准确率提升 **10-15%**

---

## 方案3: 推理时Chain-of-Thought

### 📋 目标
- 推理时让模型输出思考过程
- 特别是对于位置关系问题

### 💡 核心思路
```
原方案: 问题 → 模型 → 答案
新方案: 问题 → 模型 → 推理步骤 → 答案提取 → 最终答案
```

### 🔧 具体实施步骤

#### 步骤1: CoT Prompt设计

```python
# 修改 test_b_style_refer_215.py

def build_cot_prompt(question, images, question_type='position'):
    """
    构建Chain-of-Thought提示词

    Args:
        question_type: 'position' (位置关系), 'identification' (部件识别), 'other'
    """

    if question_type == 'position':
        prompt = f"""你是一个专利分析专家。请仔细分析图片，一步步回答下面的位置关系问题。

问题：{question}

请按照以下步骤思考：

【步骤1：识别关键部件】
首先，我需要在图中找到问题提到的部件，并记下它们的编号。

【步骤2：观察空间位置】
然后，我需要仔细观察这些部件在图中的相对位置关系。

【步骤3：确定方向关系】
接下来，我需要确定它们之间的方向关系（上下、左右、前后等）。

【步骤4：得出结论】
最后，基于以上观察，我可以给出准确的答案。

现在请开始你的分析，并在最后用【最终答案】标注你的结论："""

    elif question_type == 'identification':
        prompt = f"""你是一个专利分析专家。请仔细分析图片，一步步回答下面的部件识别问题。

问题：{question}

请按照以下步骤思考：

【步骤1：定位目标编号】
首先，我需要在图中找到问题提到的编号。

【步骤2：观察部件特征】
然后，我需要观察这个编号指向的部件的外观和特征。

【步骤3：结合上下文】
接下来，我需要结合图片的整体结构和其他信息来判断。

【步骤4：给出答案】
最后，我可以确定这个部件是什么。

现在请开始你的分析，并在最后用【最终答案】标注你的结论："""

    else:
        # 默认prompt
        prompt = question

    return prompt

def classify_question_type(question):
    """分类问题类型"""
    position_keywords = ['位置', '方向', '上方', '下方', '左侧', '右侧', '哪里', '哪个位置']
    identification_keywords = ['是什么', '什么部件', '哪个部件', '叫什么']

    if any(kw in question for kw in position_keywords):
        return 'position'
    elif any(kw in question for kw in identification_keywords):
        return 'identification'
    else:
        return 'other'
```

#### 步骤2: 答案提取

```python
def extract_final_answer(cot_response, style_examples=None):
    """
    从CoT响应中提取最终答案

    Args:
        cot_response: 包含推理过程的完整回答
        style_examples: 风格示例（用于规范化答案格式）

    Returns:
        简洁的最终答案
    """
    import re

    # 方法1: 提取【最终答案】标记的内容
    answer_match = re.search(r'【最终答案】\s*(.*?)(?:\n|$)', cot_response, re.DOTALL)
    if answer_match:
        raw_answer = answer_match.group(1).strip()
    else:
        # 如果没有标记，使用最后一句话
        sentences = cot_response.strip().split('。')
        raw_answer = sentences[-1] if sentences else cot_response

    # 方法2: 使用小模型进一步精炼答案
    if style_examples:
        refine_prompt = f"""请将下面的答案精炼为简洁的形式（20字以内）。

参考风格示例：
{style_examples[0]}
{style_examples[1]}

待精炼的答案：
{raw_answer}

精炼后的答案："""

        # 使用轻量模型快速提取
        refined = origin_vllm([{"role": "user", "content": refine_prompt}], max_tokens=50)
        return refined.strip()

    return raw_answer.strip()
```

#### 步骤3: 集成到推理流程

```python
# 修改 test_b_style_refer_215.py 的主推理循环

for idx in range(len(df_question)):
    question = df_question.loc[idx, 'question']
    document_name = df_question.loc[idx, 'document']

    # 分类问题
    question_type = classify_question_type(question)

    # 获取优化的图像输入
    image_config = get_optimized_image_input(question, document_name, idx)

    # 判断是否使用CoT
    use_cot = question_type in ['position', 'identification']

    if use_cot:
        # 使用CoT prompt
        prompt = build_cot_prompt(question, image_config['images'], question_type)

        # 设置更高的max_tokens以容纳推理过程
        max_tokens = 1024
    else:
        # 使用普通prompt
        prompt = question
        max_tokens = 512

    # 构造messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image", "image": img} for img in image_config['images']]
            ]
        }
    ]

    # 生成回答（包含推理过程）
    cot_response = origin_vllm(messages, max_tokens=max_tokens)

    # 提取最终答案
    if use_cot:
        # 获取风格示例
        similar_q_idx = get_similar_question_embedding(idx, top_k=2)
        style_examples = get_options_for_similar_answer(similar_q_idx)

        final_answer = extract_final_answer(cot_response, style_examples)
    else:
        final_answer = cot_response

    # 保存结果
    result = {
        'question': question,
        'raw_response': cot_response if use_cot else None,  # 可选：保存完整推理过程用于分析
        'answer': final_answer
    }

    results.append(result)
```

### 📊 预期效果

**推理质量**:
- 位置关系问题准确率提升 **10-15%**
- 可解释性显著提升

**性能影响**:
- 推理时间增加约 **30-50%** (因为max_tokens更大)
- 可通过仅对特定问题使用CoT来平衡

---

## 🎯 综合实施方案

### 阶段1: 快速验证 (1周)

1. **实施方案3** (CoT prompting)
   - 无需额外数据准备
   - 立即可测试效果
   - 预期提升: 8-12%

### 阶段2: 数据增强 (2周)

2. **实施方案2** (推理链生成)
   - 生成推理链数据
   - 混合训练
   - 预期提升: 额外5-8%

### 阶段3: 深度优化 (3-4周)

3. **实施方案1** (文档结构化)
   - 开发版面分析工具
   - 修改预处理和推理流程
   - 预期提升: 额外10-15%

### 累计预期效果

```
当前准确率: 82%
+ 方案3 (CoT): +10% → 90.4%
+ 方案2 (推理链): +5% → 94.9%
+ 方案1 (结构化): +10% → 100% (理论上限)
```

**实际预期**: 提升至 **90-95%** 的准确率

---

## 📝 实施检查清单

### 方案1: 文档结构化
- [ ] 开发 DocumentStructureParser 类
- [ ] 实现图片裁剪功能
- [ ] 批量解析所有文档
- [ ] 修改问题解析逻辑
- [ ] 更新推理脚本
- [ ] 测试和验证

### 方案2: 推理链生成
- [ ] 开发 ReasoningChainGenerator
- [ ] 批量生成推理链
- [ ] 合并训练数据
- [ ] 重新训练模型
- [ ] 评估效果

### 方案3: CoT推理
- [ ] 设计CoT prompts
- [ ] 实现问题分类
- [ ] 开发答案提取逻辑
- [ ] 集成到推理流程
- [ ] 测试和优化

---

## 🔬 实验建议

### 对比实验

| 实验组 | 配置 | 预期准确率 |
|--------|------|------------|
| Baseline | 当前方案 | 82% |
| Exp1 | Baseline + CoT | 90% |
| Exp2 | Baseline + 推理链训练 | 87% |
| Exp3 | Baseline + 结构化 | 92% |
| Exp4 | Exp1 + Exp2 | 93% |
| Exp5 | Exp1 + Exp2 + Exp3 | 95% |

### 评估指标

1. **整体准确率**: 所有问题的准确率
2. **位置关系准确率**: 专门针对位置关系问题
3. **推理时间**: 平均每个问题的推理时间
4. **可解释性**: 人工评估推理过程的质量

---

## 💡 其他优化建议

### 1. 多模型集成
```python
# 使用多个checkpoint投票
checkpoints = [
    'checkpoint-90',
    'checkpoint-180',
    'checkpoint-215'
]

# 对每个问题，多个模型分别推理
answers = [model(question) for model in checkpoints]

# 投票或加权融合
final_answer = vote(answers)
```

### 2. 主动学习
```python
# 对于低置信度的预测，标注并加入训练集
if confidence < 0.7:
    # 人工标注正确答案
    manual_label = get_human_annotation(question, prediction)
    # 加入训练集
    add_to_training_set(question, manual_label)
```

### 3. 知识蒸馏
```python
# 用更大的模型（teacher）训练小模型（student）
teacher_model = Qwen2.5-VL-72B
student_model = Qwen2.5-VL-32B

# 用teacher的输出（包括推理过程）训练student
```

---

## 📚 参考资料

1. **Chain-of-Thought Prompting**:
   - Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

2. **文档版面分析**:
   - LayoutLMv3, DocFormer等文档理解模型

3. **知识蒸馏**:
   - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"

4. **多模态推理**:
   - Qwen-VL官方文档和最佳实践

---

**最后更新**: 2025-11-25
**状态**: 待实施
**优先级**: 高
