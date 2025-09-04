"""
修复版Colab训练脚本
自动处理路径问题
"""

import os
import sys
import subprocess
from datetime import datetime

# 添加项目路径
project_path = "/content/timellm/Time-LLM-main"
if os.path.exists(project_path):
    sys.path.append(project_path)
    os.chdir(project_path)
    print(f"切换到项目目录: {project_path}")

def find_dataset_path(dataset_name='DEAP'):
    """自动查找数据集路径"""
    
    # 可能的数据路径列表
    possible_paths = [
        f"/content/drive/MyDrive/EEG_Data/{dataset_name}",
        f"/content/drive/MyDrive/{dataset_name}",
        f"/content/drive/MyDrive/dataset/{dataset_name}",
        f"/content/drive/MyDrive/Time-LLM-EEG/dataset/{dataset_name}",
        f"/content/drive/My Drive/EEG_Data/{dataset_name}",  # 有时是My Drive（带空格）
        f"/content/drive/My Drive/{dataset_name}",
        f"./dataset/{dataset_name}",  # 本地路径
    ]
    
    # 添加数据集特定的子目录
    if dataset_name == 'DEAP':
        extra_paths = [
            f"/content/drive/MyDrive/DEAP/data_preprocessed_python",
            f"/content/drive/MyDrive/EEG_Data/DEAP/data_preprocessed_python",
        ]
        possible_paths.extend(extra_paths)
    elif dataset_name == 'SEED':
        extra_paths = [
            f"/content/drive/MyDrive/SEED/SEED_EEG",
            f"/content/drive/MyDrive/SEED/Preprocessed_EEG",
            f"/content/drive/MyDrive/EEG_Data/SEED/Preprocessed_EEG",
        ]
        possible_paths.extend(extra_paths)
    
    # 查找存在的路径
    for path in possible_paths:
        if os.path.exists(path):
            # 检查是否包含.mat文件
            try:
                files = os.listdir(path)
                mat_files = [f for f in files if f.endswith('.mat')]
                if mat_files:
                    print(f"✓ 找到{dataset_name}数据集: {path}")
                    print(f"  包含{len(mat_files)}个.mat文件")
                    return path
            except:
                continue
    
    return None


def run_training(dataset='DEAP', custom_args=None):
    """运行训练"""
    
    # 查找数据路径
    data_path = find_dataset_path(dataset)
    
    if data_path is None:
        print(f"\n❌ 错误：找不到{dataset}数据集！")
        print("\n请检查：")
        print("1. Google Drive是否已挂载")
        print("2. 数据集是否已上传到Drive")
        print("3. 数据集路径是否正确")
        print("\n尝试过的路径：")
        print("- /content/drive/MyDrive/EEG_Data/")
        print("- /content/drive/MyDrive/dataset/")
        print("- /content/drive/MyDrive/Time-LLM-EEG/dataset/")
        return
    
    # 生成实验ID
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 构建基础参数
    base_args = {
        'is_training': 1,
        'model': 'TimeLLM',
        'task_name': 'classification',
        'data': dataset,
        'root_path': data_path,
        'data_path': 'data.csv',
        'features': 'M',
        'target': 'label',
        'checkpoints': './checkpoints/',
        'seq_len': 256,
        'label_len': 0,
        'pred_len': 0,
        'd_model': 32,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 64,
        'dropout': 0.1,
        'embed': 'timeF',
        'freq': 'h',
        'factor': 1,
        'activation': 'gelu',
        'distil': True,
        'moving_avg': 25,
        'num_workers': 0,
        'itr': 1,
        'train_epochs': 50,
        'patience': 15,
        'des': 'enhanced',
        'lradj': 'COS',
        'pct_start': 0.2,
        'eval_batch_size': 8,
        'llm_model': 'GPT2',
        'llm_dim': 768,
        'llm_layers': 6,
        'patch_len': 16,
        'stride': 8,
        'temperature': 10,
        'percent': 100,
        'seasonal_patterns': 'Monthly',
        'loss': 'MSE',
        'align_epochs': 10,
        'optimizer': 'adamw',
        'weight_decay': 0.0001,
        'scheduler_type': 'cosine',
        'grad_clip': 1.0,
    }
    
    # 数据集特定参数
    if dataset == 'DEAP':
        dataset_args = {
            'enc_in': 32,
            'dec_in': 32,
            'c_out': 32,
            'num_class': 2,
            'batch_size': 16,
            'learning_rate': 0.0001,
            'loss_type': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'warmup_epochs': 5,
            'model_id': f'TimeLLM_DEAP_focal_{current_time}',
            'model_comment': 'DEAP_focal'
        }
    else:  # SEED
        dataset_args = {
            'enc_in': 62,
            'dec_in': 62,
            'c_out': 62,
            'num_class': 3,
            'batch_size': 8,
            'learning_rate': 0.00005,
            'loss_type': 'weighted_ce',
            'warmup_epochs': 10,
            'model_id': f'TimeLLM_SEED_wce_{current_time}',
            'model_comment': 'SEED_weighted_ce'
        }
    
    # 合并参数
    all_args = {**base_args, **dataset_args}
    
    # 如果有自定义参数，覆盖默认值
    if custom_args:
        all_args.update(custom_args)
    
    # 构建命令
    cmd = ["python", "run_main_enhanced.py"]
    for key, value in all_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    # 打印配置
    print("\n" + "=" * 70)
    print(f"{dataset} 训练配置")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"模型ID: {all_args['model_id']}")
    print(f"损失函数: {all_args['loss_type']}")
    print(f"批量大小: {all_args['batch_size']}")
    print(f"学习率: {all_args['learning_rate']}")
    print(f"训练轮数: {all_args['train_epochs']}")
    
    # 执行训练
    print("\n开始训练...")
    print("命令:", " ".join(cmd[:5]) + " ...")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败: {e}")
    except KeyboardInterrupt:
        print("\n训练被中断")


# 快速训练函数
def train_deap_focal():
    """DEAP数据集 - Focal Loss训练"""
    run_training('DEAP', {
        'loss_type': 'focal',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0
    })


def train_seed_weighted():
    """SEED数据集 - 加权CE训练"""
    run_training('SEED', {
        'loss_type': 'weighted_ce'
    })


def train_quick_test(dataset='DEAP'):
    """快速测试（少量epoch）"""
    run_training(dataset, {
        'train_epochs': 5,
        'batch_size': 4,
        'model_id': f'TimeLLM_{dataset}_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Time-LLM EEG训练脚本（Colab版）')
    parser.add_argument('--dataset', type=str, default='DEAP', choices=['DEAP', 'SEED'])
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'test', 'custom'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    
    # 自定义参数
    custom_args = {}
    if args.epochs:
        custom_args['train_epochs'] = args.epochs
    if args.batch_size:
        custom_args['batch_size'] = args.batch_size
    if args.lr:
        custom_args['learning_rate'] = args.lr
    
    # 根据模式运行
    if args.mode == 'test':
        train_quick_test(args.dataset)
    else:
        run_training(args.dataset, custom_args if custom_args else None)
