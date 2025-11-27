#!/bin/bash
###############################################################################
# 环境检查脚本
# 功能: 在运行前检查所有必要的依赖和文件
###############################################################################

echo "=========================================="
echo "CCKS 2025 环境检查工具"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"

# 统计
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

check_pass() {
    echo -e "$PASS $1"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
}

check_fail() {
    echo -e "$FAIL $1"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

check_warn() {
    echo -e "$WARN $1"
    WARNINGS=$((WARNINGS + 1))
}

# 1. 检查Python版本
echo "1. 检查Python环境"
echo "---"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        check_pass "Python版本: $PYTHON_VERSION"
    else
        check_fail "Python版本过低: $PYTHON_VERSION (需要 >= 3.10)"
    fi
else
    check_fail "未安装Python"
fi
echo ""

# 2. 检查Python依赖
echo "2. 检查Python依赖包"
echo "---"
REQUIRED_PACKAGES=("torch" "transformers" "vllm" "swift" "numpy" "pandas" "PIL" "fitz")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "未知版本")
        check_pass "$package ($VERSION)"
    else
        check_fail "$package (未安装)"
    fi
done
echo ""

# 3. 检查CUDA和GPU
echo "3. 检查CUDA和GPU"
echo "---"
if command -v nvidia-smi &> /dev/null; then
    check_pass "nvidia-smi 可用"

    # 检查GPU数量
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    check_pass "检测到 $GPU_COUNT 个GPU"

    # 检查CUDA版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    check_pass "CUDA版本: $CUDA_VERSION"

    # 检查每个GPU的状态
    echo ""
    echo "GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    while IFS=, read -r idx name mem_used mem_total util; do
        mem_used_gb=$(echo "$mem_used" | awk '{print int($1/1024)}')
        mem_total_gb=$(echo "$mem_total" | awk '{print int($1/1024)}')
        mem_free_gb=$((mem_total_gb - mem_used_gb))

        if [ "$mem_free_gb" -gt 50 ]; then
            echo -e "  $PASS GPU $idx: $name - ${mem_free_gb}GB 可用 (使用率: $util)"
        elif [ "$mem_free_gb" -gt 20 ]; then
            echo -e "  $WARN GPU $idx: $name - ${mem_free_gb}GB 可用 (使用率: $util)"
        else
            echo -e "  $FAIL GPU $idx: $name - ${mem_free_gb}GB 可用 (使用率: $util) [显存不足]"
        fi
    done
else
    check_fail "nvidia-smi 不可用"
fi
echo ""

# 4. 检查项目路径配置
echo "4. 检查项目路径"
echo "---"
PROJECT_ROOT=${PROJECT_ROOT:-"/data/coding"}

if [ -d "$PROJECT_ROOT" ]; then
    check_pass "项目根目录存在: $PROJECT_ROOT"
else
    check_fail "项目根目录不存在: $PROJECT_ROOT"
fi

# 检查模型路径
MODEL_PATHS=(
    "$PROJECT_ROOT/llm_model/iic/gme-Qwen2-VL-7B-Instruct"
    "$PROJECT_ROOT/llm_model/Qwen/Qwen2___5-VL-32B-Instruct"
)

for model_path in "${MODEL_PATHS[@]}"; do
    if [ -d "$model_path" ]; then
        size=$(du -sh "$model_path" 2>/dev/null | awk '{print $1}')
        check_pass "模型存在: $(basename $model_path) ($size)"
    else
        check_fail "模型不存在: $model_path"
    fi
done

# 检查数据路径
DATA_PATHS=(
    "$PROJECT_ROOT/patent_b/train/documents"
    "$PROJECT_ROOT/patent_b/test/documents"
)

for data_path in "${DATA_PATHS[@]}"; do
    if [ -d "$data_path" ]; then
        pdf_count=$(ls "$data_path"/*.pdf 2>/dev/null | wc -l)
        check_pass "数据目录存在: $(basename $(dirname $data_path)) ($pdf_count 个PDF)"
    else
        check_fail "数据目录不存在: $data_path"
    fi
done
echo ""

# 5. 检查磁盘空间
echo "5. 检查磁盘空间"
echo "---"
if [ -d "$PROJECT_ROOT" ]; then
    DISK_AVAIL=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    DISK_AVAIL_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print int($4)}')

    if [ "$DISK_AVAIL_GB" -gt 200 ]; then
        check_pass "可用磁盘空间: $DISK_AVAIL"
    elif [ "$DISK_AVAIL_GB" -gt 100 ]; then
        check_warn "可用磁盘空间: $DISK_AVAIL (建议 > 200GB)"
    else
        check_fail "可用磁盘空间不足: $DISK_AVAIL (需要 > 200GB)"
    fi
else
    check_warn "无法检查磁盘空间（目录不存在）"
fi
echo ""

# 6. 检查运行脚本
echo "6. 检查运行脚本"
echo "---"
SCRIPT_DIR="scripts"
SCRIPTS=(
    "00_setup_paths.sh"
    "01_preprocess.sh"
    "02_train.sh"
    "03_inference.sh"
)

if [ -d "$SCRIPT_DIR" ]; then
    for script in "${SCRIPTS[@]}"; do
        script_path="$SCRIPT_DIR/$script"
        if [ -f "$script_path" ]; then
            if [ -x "$script_path" ]; then
                check_pass "$script (可执行)"
            else
                check_warn "$script (不可执行，请运行: chmod +x $script_path)"
            fi
        else
            check_fail "$script 不存在"
        fi
    done
else
    check_fail "scripts目录不存在"
fi
echo ""

# 7. 检查tmux
echo "7. 检查tmux"
echo "---"
if command -v tmux &> /dev/null; then
    TMUX_VERSION=$(tmux -V | awk '{print $2}')
    check_pass "tmux 已安装 (版本: $TMUX_VERSION)"

    # 检查是否有运行中的会话
    TMUX_SESSIONS=$(tmux ls 2>/dev/null | wc -l)
    if [ "$TMUX_SESSIONS" -gt 0 ]; then
        check_pass "检测到 $TMUX_SESSIONS 个tmux会话"
        tmux ls 2>/dev/null | while read line; do
            echo "    $line"
        done
    fi
else
    check_warn "tmux 未安装 (建议安装: apt-get install tmux)"
fi
echo ""

# 8. 总结
echo "=========================================="
echo "检查结果汇总"
echo "=========================================="
echo "总检查项: $TOTAL_CHECKS"
echo -e "通过: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "失败: ${RED}$FAILED_CHECKS${NC}"
echo -e "警告: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo -e "${GREEN}✓ 环境检查通过！可以开始复现。${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 配置路径: bash scripts/00_setup_paths.sh /data/coding"
    echo "  2. 启动tmux: tmux new -s ccks2025"
    echo "  3. 运行预处理: bash scripts/01_preprocess.sh"
    exit 0
elif [ "$FAILED_CHECKS" -le 2 ]; then
    echo -e "${YELLOW}⚠ 部分检查失败，但可能不影响运行。${NC}"
    echo "请根据上述错误进行修复。"
    exit 1
else
    echo -e "${RED}✗ 环境检查失败！请先解决上述问题。${NC}"
    echo ""
    echo "常见问题解决:"
    echo "  - 安装依赖: pip install ms-swift vllm transformers torch PyMuPDF numpy pandas"
    echo "  - 下载模型: 参考 REPRODUCTION_GUIDE.md 步骤0.3"
    echo "  - 准备数据: 参考 REPRODUCTION_GUIDE.md 步骤0.4"
    exit 1
fi