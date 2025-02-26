#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time

import csv
import shutil

def organize_images(csv_path, source_base, target_base):
    """
    根据 dicom_info.csv 文件中:
        [image_path, Laterality, PatientOrientation, SeriesDescription] 四列，
    进行如下操作:

      1. 只处理 SeriesDescription == 'full mammogram images' 的行，其它(ROI mask images, cropped images)跳过；
      2. 如果 image_path 中带有 'CBIS-DDSM/jpeg/' 前缀，则去掉该前缀；
      3. 仅处理“母文件夹中仅包含 1 个 .jpg/.jpeg 文件”的图像；
      4. 符合条件的图像根据 (Laterality, PatientOrientation) 分别放到
         [LEFT_CC, LEFT_MLO, RIGHT_CC, RIGHT_MLO]，否则归入 OTHER；
      5. 统计成功复制(copied_count)、跳过(skipped_count)及 OTHER 文件夹计数(other_count)。

    :param csv_path:    dicom_info.csv 文件路径
    :param source_base: JPEG 文件根目录
    :param target_base: 分类后图像的输出根目录
    """

    if not os.path.exists(target_base):
        os.makedirs(target_base)
    
    copied_count = 0
    skipped_count = 0
    other_count = 0

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 先检查 SeriesDescription
            series_desc = row.get('SeriesDescription', '').strip().lower()
            if series_desc != 'full mammogram images':
                # 如果不是 full mammogram images，则跳过
                skipped_count += 1
                continue

            # 读取 CSV 中其余字段
            raw_image_path = row.get('image_path', '').strip()
            laterality = row.get('Laterality', '').strip().upper()
            orientation = row.get('PatientOrientation', '').strip().upper()

            if not raw_image_path or not laterality or not orientation:
                print(f"[警告] 缺少必要字段，跳过: {row}")
                skipped_count += 1
                continue

            # 若路径里包含多余的 "CBIS-DDSM/jpeg/" 前缀，自动去掉
            prefix = "CBIS-DDSM/jpeg/"
            if raw_image_path.startswith(prefix):
                raw_image_path = raw_image_path[len(prefix):]

            # 拼接得到文件绝对路径
            source_image_path = os.path.join(source_base, raw_image_path)
            if not os.path.isfile(source_image_path):
                print(f"[警告] 源文件不存在，跳过: {source_image_path}")
                skipped_count += 1
                continue

            # 只处理“母文件夹中只包含 1 个 .jpg/.jpeg 文件”
            parent_dir = os.path.dirname(source_image_path)
            if not os.path.isdir(parent_dir):
                print(f"[警告] 上级目录不存在，跳过: {parent_dir}")
                skipped_count += 1
                continue

            all_jpegs = [
                fname for fname in os.listdir(parent_dir)
                if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg')
            ]
            if len(all_jpegs) != 1:
                print(f"[提示] 文件夹 {parent_dir} 中有 {len(all_jpegs)} 个 JPG/JPEG 文件，跳过.")
                skipped_count += 1
                continue

            # 分类逻辑
            if (laterality.startswith('L')) and (orientation == 'CC'):
                sub_folder = 'LEFT_CC'
            elif (laterality.startswith('L')) and (orientation == 'MLO'):
                sub_folder = 'LEFT_MLO'
            elif (laterality.startswith('R')) and (orientation == 'CC'):
                sub_folder = 'RIGHT_CC'
            elif (laterality.startswith('R')) and (orientation == 'MLO'):
                sub_folder = 'RIGHT_MLO'
            else:
                sub_folder = 'OTHER'
                other_count += 1

            # 拷贝文件到目标子文件夹
            target_dir = os.path.join(target_base, sub_folder)
            os.makedirs(target_dir, exist_ok=True)

            try:
                shutil.copy2(source_image_path, target_dir)
                copied_count += 1
            except Exception as e:
                print(f"[错误] 无法拷贝 {source_image_path} 至 {target_dir}: {e}")
                skipped_count += 1
    
    print("== 处理结果统计 ==")
    print(f"成功复制的图像数量: {copied_count}")
    print(f"跳过的图像数量:     {skipped_count}")
    print(f"OTHER 文件夹计数:   {other_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Organize CBIS-DDSM images (only full mammogram).")
    parser.add_argument('--csv_path', type=str, required=False, default='/burg/biostats/users/yl5465/DM_BC/dataset/CBIS-DDSM/csv/dicom_info.csv',
                        help='Path to the dicom_info.csv file.')
    parser.add_argument('--source_base', type=str, required=False, default='/burg/biostats/users/yl5465/DM_BC/dataset/CBIS-DDSM/jpeg',
                        help='Root directory of the source JPEG images.')
    parser.add_argument('--target_base', type=str, required=False, default='/burg/biostats/users/yl5465/DM_BC/dataset/organized_images',
                        help='Output directory to save the organized images.')
    return parser.parse_args()

def main():
    args = parse_args()
    organize_images(
        csv_path=args.csv_path,
        source_base=args.source_base,
        target_base=args.target_base
    )
    print("done!")

if __name__ == "__main__":
    main()
