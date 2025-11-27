# 快速启动指南

**适用于**: 8卡A800服务器（GPU 5,6,7正在使用的情况）

## 5分钟快速开始

### 1. 配置Conda环境（一键完成）⭐
```bash
cd /path/to/project/root

# 自动配置环境（推荐）
bash setup_conda_env.sh

# 或手动配置
conda env create -f environment.yml
conda activate ccks2025_pdf_qa

# 或使用pip
pip install -r requirements.txt
```

### 2. 进入工作目录
```bash
cd ccks2025_pdf_multimodal/round_b
```

### 3. 配置路径
```bash
bash scripts/00_setup_paths.sh /data/coding
```

### 4. 启动tmux会话
```bash
tmux new -s ccks2025

# 如果使用conda，确保环境已激活
conda activate ccks2025_pdf_qa
```

### 5. 运行预处理（6-8小时）
```bash
bash scripts/01_preprocess.sh
```

按 `Ctrl+B, D` 分离会话，让其后台运行。

### 6. 构造训练集
```bash
# 重新连接tmux
tmux attach -t ccks2025

# 确保环境激活
conda activate ccks2025_pdf_qa  # 如使用conda

# 启动Jupyter
jupyter notebook --no-browser --port=8888

# 打开并运行: finetune训练集构造_v2.ipynb
```

### 7. 训练模型（6-8小时）
```bash
bash scripts/02_train.sh
```

### 8. 生成结果（2-3小时）
```bash
bash scripts/03_inference.sh
```

完成！结果在 `test_b_style_infer_if_need_ck215.jsonl`

---

## GPU使用分配

根据你的GPU占用情况，脚本已优化为：

| 步骤 | GPU | 说明 |
|------|-----|------|
| 预处理 | GPU 1 | 最空闲（74GB可用）|
| 训练 | GPU 0,1,2,3,4 | 5张卡（避开5,6,7）|
| 推理 | GPU 0,1,2,3 | 4张卡 |

---

## 检查清单

开始前确认：
- [ ] 已配置Conda环境: `bash setup_conda_env.sh` 或 `pip install -r requirements.txt`
- [ ] 已下载模型: GME-Qwen2-VL-7B + Qwen2.5-VL-32B (~78GB)
- [ ] 已下载数据: 训练集 + 测试集 PDF文件
- [ ] 已启动tmux: `tmux new -s ccks2025`
- [ ] 磁盘空间充足: 至少200GB可用
- [ ] 如使用Conda: `conda activate ccks2025_pdf_qa`

---

## 监控命令

```bash
# 查看GPU状态
watch -n 5 nvidia-smi

# 查看日志
tail -f preprocess.log  # 或 train.log, inference.log

# 重新连接tmux
tmux attach -t ccks2025

# 查看进度（预处理）
ls -lh *_vectors.npy

# 查看进度（训练）
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*
```

---

## 预期时间线

```
00:00 - 开始预处理
06:00 - 预处理完成，构造训练集 (+30分钟)
06:30 - 开始训练
14:30 - 训练完成，开始推理
17:00 - 完成！
```

**总计**: 约17小时（可在tmux后台运行）

---

## 快速问题解决

### OOM错误
```bash
# 降低分辨率
export MAX_PIXELS=819200
```

### 路径错误
```bash
# 重新配置
bash scripts/00_setup_paths.sh
```

### 训练中断
```bash
# SWIFT会自动从最新checkpoint恢复
bash scripts/02_train.sh
```

---

## 一键运行（实验性）

⚠️ **仅适用于确认所有配置正确的情况**

```bash
cd /path/to/ccks2025_pdf_multimodal/round_b
tmux new -s ccks2025

# 运行完整流程
bash scripts/00_setup_paths.sh /data/coding && \
bash scripts/01_preprocess.sh && \
echo "=== 请运行Jupyter notebook构造训练集 ===" && \
read -p "完成后按Enter继续..." && \
bash scripts/02_train.sh && \
bash scripts/03_inference.sh && \
echo "=== 完成！结果文件: test_b_style_infer_if_need_ck215.jsonl ==="

# Ctrl+B, D 分离会话
```

---

## 获取帮助

- 详细指南: `REPRODUCTION_GUIDE.md`
- 脚本说明: `scripts/README.md`
- 项目文档: `CLAUDE.md`
- 技术分析: `技术分析报告.md`