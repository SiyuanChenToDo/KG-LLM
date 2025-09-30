#!/usr/bin/env python3
# excel_to_csv.py
# 用法：python excel_to_csv.py selected_papers_final.xlsx

import pandas as pd
import sys
import os
from pathlib import Path

def excel_to_csv(xlsx_path, output_dir=None):
    """
    将 xlsx 文件的所有工作表分别另存为 csv 文件。
    参数
    ----
    xlsx_path : str
        输入的 .xlsx 文件路径
    output_dir : str, optional
        输出目录；缺省则在 xlsx 同级目录下新建同名文件夹
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    if output_dir is None:
        output_dir = xlsx_path.with_suffix('')  # 同名文件夹
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有工作表
    excel_file = pd.ExcelFile(xlsx_path)
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)  # dtype=str 防止长数字被截断
        csv_path = output_dir / f"{sheet}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig 兼容 Excel
        print(f"已生成 {csv_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python excel_to_csv.py <xxx.xlsx> [输出目录]")
        sys.exit(1)
    xlsx_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    excel_to_csv(xlsx_file, out_dir)