#!/usr/bin/env python3
"""从 test.parquet 中抽取前 N 条记录并保存为 val.parquet。"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 test.parquet 抽取前 N 条样本保存为 val.parquet"
    )
    parser.add_argument(
        "--input",
        default="data/qrecc_plan_b/test.parquet",
        help="源 parquet 文件路径，默认 data/qrecc_plan_b/test.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/qrecc_plan_b/val.parquet",
        help="输出 parquet 文件路径，默认 data/qrecc_plan_b/val.parquet",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=800,
        help="需要抽取的样本数，默认 800",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.input)
    if not src_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{src_path}")

    df = pd.read_parquet(src_path)
    if args.num <= 0:
        raise ValueError("抽样数量必须为正整数")

    subset = df.iloc[: args.num].reset_index(drop=True)

    dst_path = Path(args.output)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(dst_path, index=False)
    print(f"已将前 {len(subset)} 条记录保存到 {dst_path}")


if __name__ == "__main__":
    main()
