#!/bin/bash
#
# Quick command to switch training config between E5 and BM25
#

echo "Which retrieval method are you using?"
echo "1) E5 (GPU-based, 3 GPUs for training)"
echo "2) BM25 (CPU-based, 4 GPUs for training)"
read -p "Enter choice [1-2]: " choice

if [ "$choice" = "1" ]; then
    # E5: Use GPU 5,6,7 for training (GPU 4 for E5 server)
    sed -i 's/export CUDA_VISIBLE_DEVICES=.*/export CUDA_VISIBLE_DEVICES=5,6,7/' configs/qrecc/train_qrecc_ppo_plan_b.sh
    sed -i 's/trainer.n_gpus_per_node=.*/trainer.n_gpus_per_node=3 \\/' configs/qrecc/train_qrecc_ppo_plan_b.sh
    echo "✓ Configured for E5 (3 GPUs: 5,6,7)"
elif [ "$choice" = "2" ]; then
    # BM25: Use GPU 4,5,6,7 for training (BM25 uses CPU only)
    sed -i 's/export CUDA_VISIBLE_DEVICES=.*/export CUDA_VISIBLE_DEVICES=4,5,6,7/' configs/qrecc/train_qrecc_ppo_plan_b.sh
    sed -i 's/trainer.n_gpus_per_node=.*/trainer.n_gpus_per_node=4 \\/' configs/qrecc/train_qrecc_ppo_plan_b.sh
    echo "✓ Configured for BM25 (4 GPUs: 4,5,6,7)"
else
    echo "Invalid choice"
fi