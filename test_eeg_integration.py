"""
测试EEG数据集是否成功集成到 Time-LLM 中
1. 支持 DEAP 与 SEED 的加载测试（train/val/test）
2. 验证自动被试划分逻辑是否生效
3. 打印数据 shape、标签分布等，确保输入格式对模型友好
"""

import sys
import os
import torch
import numpy as np
from collections import Counter

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider


class Args:
    def __init__(self):
        # 通用参数
        self.batch_size = 32
        self.num_workers = 0
        self.seq_len = 256
        self.pred_len = 0
        self.label_len = 0
        self.embed = 'timeF'
        self.freq = 'h'
        self.features = 'M'
        self.target = 'OT'
        self.seasonal_patterns = 'Monthly'

        # EEG 专属参数（默认值）
        self.data = 'DEAP'
        self.root_path = ''
        self.num_class = 2
        self.classification_type = 'valence'
        self.overlap = 128
        self.normalize = True
        self.filter_freq = None
        self.sampling_rate = 128
        self.subject_list = None  # 仅在 train 阶段指定


def print_batch_info(loader, desc=""):
    for batch in loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        print(f"\n[{desc}] 一个批次样本信息:")
        print(f"  - batch_x: {batch_x.shape} (batch_size, seq_len, n_channels)")
        print(f"  - batch_y: {batch_y.shape} (batch_size,)")
        print(f"  - 标签示例: {batch_y[:5].tolist()}")
        break


def summarize_labels(labels):
    counter = Counter(labels)
    return dict(counter)


def test_dataset(dataset_name):
    args = Args()
    args.data = dataset_name

    if dataset_name == 'DEAP':
        args.root_path = r"D:\\文件\\文件\\HKU\\Dissertation\\dataset\\DEAP\\data_preprocessed_python"
        args.num_class = 2
        args.sampling_rate = 128

    elif dataset_name == 'SEED':
        args.root_path = r"D:\\文件\\文件\\HKU\\Dissertation\\dataset\\SEED\\SEED\\Preprocessed_EEG"
        args.num_class = 3
        args.sampling_rate = 200

    else:
        print(f"不支持的数据集: {dataset_name}")
        return

    print(f"\n{'='*70}\n测试 {dataset_name} 数据集\n{'='*70}")

    try:
        # 训练集（指定部分被试）
        args.subject_list = ['s01'] if dataset_name == 'DEAP' else ['1_20131027.mat']
        train_data, train_loader = data_provider(args, 'train')
        print(f"✓ Train 集大小: {len(train_data)}，批次数: {len(train_loader)}")
        print(f"✓ 标签分布: {summarize_labels(train_data.labels)}")
        print_batch_info(train_loader, "Train")

        # 验证集（使用默认划分）
        args.subject_list = None
        val_data, val_loader = data_provider(args, 'val')
        print(f"✓ Val 集大小: {len(val_data)}，批次数: {len(val_loader)}")
        print(f"✓ 标签分布: {summarize_labels(val_data.labels)}")
        print_batch_info(val_loader, "Val")

        # 测试集
        test_data, test_loader = data_provider(args, 'test')
        print(f"✓ Test 集大小: {len(test_data)}，批次数: {len(test_loader)}")
        print(f"✓ 标签分布: {summarize_labels(test_data.labels)}")
        print_batch_info(test_loader, "Test")

        print(f"\n🎯 {dataset_name} 数据集测试完成！\n")

    except Exception as e:
        print(f"[ERROR] 加载 {dataset_name} 数据集失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("开始测试 EEG 数据加载与划分逻辑\n")
    test_dataset('DEAP')
    test_dataset('SEED')


if __name__ == '__main__':
    main()
