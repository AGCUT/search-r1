# CCKS 2025 专利问答项目 - 8卡A800复现指南

本仓库提供了在8卡A800服务器上完整复现CCKS 2025专利问答多模态竞赛的详细指南和自动化脚本。

## 📋 当前服务器配置

**GPU状态** (2025-11-25):
- **GPU 0**: 67.9GB/80GB 已使用 → 可用约14GB
- **GPU 1**: 7.5GB/80GB 已使用 → ✅ **可用约74GB (最空闲)**
- **GPU 2**: 40.8GB/80GB 已使用 → 可用约41GB
- **GPU 3**: 24.5GB/80GB 已使用 → 可用约57GB
- **GPU 4**: 65.5GB/80GB 已使用 → 可用约16GB
- **GPU 5-7**: ❌ **100%使用中，请勿使用**

**GPU使用策略**:
```
预处理阶段: GPU 1 (单卡，最空闲)
训练阶段:   GPU 0,1,2,3,4 (5卡并行)
推理阶段:   GPU 0,1,2,3 (4卡并行)
```

## 🚀 快速开始 (4步走)

### 步骤1: 配置Conda环境 ⭐
```bash
# 一键自动配置（推荐）
bash setup_conda_env.sh

# 激活环境
conda activate ccks2025_pdf_qa

# 或手动配置
conda env create -f environment.yml
conda activate ccks2025_pdf_qa

# 或使用pip
pip install -r requirements.txt
```

**详细说明**: 查看 `CONDA_SETUP.md` 获取完整配置指南

### 步骤2: 环境检查
```bash
cd ccks2025_pdf_multimodal/round_b
bash scripts/check_environment.sh
```

### 步骤3: 配置路径
```bash
bash scripts/00_setup_paths.sh /data/coding
```

### 步骤4: 开始复现
```bash
# 在tmux会话中运行
tmux new -s ccks2025

# 确保环境激活
conda activate ccks2025_pdf_qa  # 如使用conda

# 预处理 (6-8小时)
bash scripts/01_preprocess.sh

# 构造训练集 (手动运行notebook)
jupyter notebook finetune训练集构造_v2.ipynb

# 训练 (6-8小时)
bash scripts/02_train.sh

# 推理 (2-3小时)
bash scripts/03_inference.sh
```

## 📚 文档索引

### 核心文档
| 文档 | 用途 | 详细程度 |
|------|------|----------|
| **QUICKSTART.md** | 5分钟快速启动 | ⭐ 极简 |
| **REPRODUCTION_GUIDE.md** | 完整复现步骤 | ⭐⭐⭐⭐⭐ 详细 |
| **scripts/README.md** | 脚本使用说明 | ⭐⭐⭐ 中等 |

### 技术文档
| 文档 | 用途 |
|------|------|
| **CLAUDE.md** | 项目架构和技术细节 |
| **技术分析报告.md** | 深度技术分析和改进建议 |
| **README.md** | 原始项目说明 |

### 脚本文件
| 脚本 | 功能 | GPU | 耗时 |
|------|------|-----|------|
| `check_environment.sh` | 环境检查 | 无 | 1分钟 |
| `00_setup_paths.sh` | 路径配置 | 无 | 1分钟 |
| `01_preprocess.sh` | 数据预处理 | GPU 1 | 6-8小时 |
| `02_train.sh` | 模型训练 | GPU 0-4 | 6-8小时 |
| `03_inference.sh` | 模型推理 | GPU 0-3 | 2-3小时 |

## 📊 完整流程时间线

```
时间轴          阶段              GPU使用        状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
00:00          环境准备           无            ⚙️ 安装依赖、下载模型
01:00          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✓ 准备完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
01:00          数据预处理          GPU 1         🔄 PDF转图像+向量化
07:00          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✓ 预处理完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
07:00          构造训练集          无            📝 Jupyter notebook
07:30          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✓ 训练集生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
07:30          模型训练            GPU 0-4       🎓 LoRA微调32B模型
15:30          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✓ 训练完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15:30          模型推理            GPU 0-3       🔮 生成测试集答案
18:00          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✓ 推理完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总耗时: ~17小时 (可在tmux后台运行)
```

## 🔧 关键参数调整

由于GPU 5-7正在使用，脚本已根据实际情况优化：

### 原始配置 vs 当前配置

| 参数 | 原始（8卡） | 当前（5卡） | 影响 |
|------|-----------|-----------|------|
| 训练GPU数量 | 8 | 5 | 训练时间可能多1-2小时 |
| 有效batch size | 128 | 80 | 略微降低，影响较小 |
| 梯度累积步数 | 16 | 16 | 保持不变 |
| 预计训练时间 | 6小时 | 7-8小时 | 增加约15-30% |

### 如果遇到显存不足 (OOM)

**方案1: 降低图像分辨率**
```bash
# 编辑训练脚本
export MAX_PIXELS=819200  # 从1229312降低到819200
```

**方案2: 使用更少GPU**
```bash
# 编辑训练脚本
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张而非5张
```

**方案3: 增加梯度累积**
```bash
# 编辑训练脚本，修改参数
--gradient_accumulation_steps 32  # 从16增加到32
```

## 📦 所需资源

### 模型下载 (~78GB)
- GME-Qwen2-VL-7B-Instruct: ~14GB
- Qwen2.5-VL-32B-Instruct: ~64GB

### 数据集
- 训练集: ~1000个PDF，约20000页
- 测试集: ~800个PDF，约15000页

### 磁盘空间需求
- 模型: 78GB
- 数据: 20GB
- 中间文件: 50GB
- 输出: 10GB
- **总计**: ~200GB (建议预留300GB)

### 显存需求
- 预处理: 30GB/卡
- 训练: 60GB/卡 × 5卡
- 推理: 40GB/卡 × 4卡

## 🐛 常见问题

### Q1: 如何检查当前进度？

**预处理阶段**:
```bash
ls -lh *_vectors.npy  # 查看生成的向量文件
tail -f preprocess.log  # 查看实时日志
```

**训练阶段**:
```bash
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*  # 查看checkpoints
tail -f train.log  # 查看训练日志
watch -n 5 nvidia-smi  # 监控GPU使用
```

**推理阶段**:
```bash
wc -l test_b_style_infer_if_need_ck215.jsonl  # 查看已生成的结果数
tail -f inference.log  # 查看推理日志
```

### Q2: tmux会话意外断开怎么办？

```bash
# 查看所有会话
tmux ls

# 重新连接
tmux attach -t ccks2025

# 如果会话不存在，程序可能已结束，检查日志
tail -f train.log
```

### Q3: 路径配置错误怎么办？

```bash
# 重新运行路径配置脚本
bash scripts/00_setup_paths.sh /your/actual/path

# 手动检查哪些文件需要修改
grep -r "/data/coding" *.py
```

### Q4: 训练中断如何恢复？

```bash
# SWIFT支持自动从checkpoint恢复
# 直接重新运行训练脚本即可
bash scripts/02_train.sh

# 或手动指定checkpoint
# 编辑02_train.sh，添加:
--resume_from_checkpoint /data/coding/lora_qwen25_vl_32b_b/checkpoint-xxx
```

### Q5: 如何在GPU空闲后使用全部8张GPU？

```bash
# 等GPU 5-7空闲后，编辑训练脚本
# 修改 02_train.sh:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8张GPU

# 同时可以降低梯度累积以保持相同的有效batch size
--gradient_accumulation_steps 16  # 保持不变，或调整为10
```

## 📈 性能优化建议

### 1. 加速预处理
- 使用多进程并行转换PDF
- 预先过滤空白页面
- 使用SSD存储中间文件

### 2. 加速训练
- 使用DeepSpeed ZeRO优化
- 混合精度训练（已启用）
- 增大有效batch size

### 3. 加速推理
- 增加批处理大小 (`max_num_seqs=4`)
- 使用INT8量化
- 启用speculative decoding

## 🎯 验证结果

### 检查预处理输出
```bash
# 应该看到6个文件
ls -lh *_vectors.npy *.csv

# 验证向量维度
python -c "import numpy as np; print(np.load('train_b_pdf_img_vectors.npy').shape)"
# 输出应该是: (页面数, 3584)
```

### 检查训练输出
```bash
# 查看checkpoints
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*

# 查看训练日志中的loss
grep "loss" train.log | tail -n 20
```

### 检查推理输出
```bash
# 结果文件行数应该等于测试集问题数
wc -l test_b_style_infer_if_need_ck215.jsonl

# 查看结果格式
head -n 1 test_b_style_infer_if_need_ck215.jsonl | python -m json.tool
```

## 📞 获取帮助

1. **查看详细文档**:
   - 完整步骤: `REPRODUCTION_GUIDE.md`
   - 脚本说明: `scripts/README.md`
   - 快速开始: `QUICKSTART.md`

2. **运行环境检查**:
   ```bash
   bash scripts/check_environment.sh
   ```

3. **参考原始项目**:
   - 竞赛页面: https://tianchi.aliyun.com/competition/entrance/532357/information
   - 技术博客: https://zhuanlan.zhihu.com/p/1937478905532506972
   - 数据分享: https://pan.baidu.com/s/10fYbAaSNf-Nv7x1jsMiqzw (提取码: r96k)

## ✅ 检查清单

开始复现前请确认：

- [ ] 已查看GPU状态，确认可用GPU
- [ ] 已运行环境检查脚本 (`check_environment.sh`)
- [ ] 已安装所有Python依赖
- [ ] 已下载GME和Qwen2.5-VL-32B模型
- [ ] 已下载并解压训练/测试数据集
- [ ] 已配置正确的路径 (`00_setup_paths.sh`)
- [ ] 已启动tmux会话
- [ ] 磁盘空间充足 (>200GB可用)
- [ ] 已阅读 REPRODUCTION_GUIDE.md

## 🎉 预期结果

如果一切顺利，你将得到：

1. **中间文件**:
   - 训练集和测试集的向量文件 (~500MB)
   - 微调后的模型checkpoints (~5GB)

2. **最终结果**:
   - `test_b_style_infer_if_need_ck215.jsonl` - 测试集预测答案
   - 可直接提交到竞赛平台

3. **时间消耗**:
   - 总计约17小时（可后台运行）
   - 预处理: 6-8小时
   - 训练: 6-8小时
   - 推理: 2-3小时

---

**开始复现**: `bash scripts/check_environment.sh && bash scripts/00_setup_paths.sh`

**遇到问题**: 请查看 REPRODUCTION_GUIDE.md 的"常见问题排查"章节