    """
增强版EEG情绪分类训练脚本
包含所有训练辅助模块的功能
"""

import os
import sys
import argparse
from datetime import datetime

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='TimeLLM for EEG Emotion Classification (Enhanced)')
    
    # 基本配置
    parser.add_argument('--is_training', type=int, default=1, help='是否训练')
    parser.add_argument('--model_id', type=str, default='test', help='模型ID')
    parser.add_argument('--model', type=str, default='TimeLLM', help='模型名称')
    
    # 任务配置
    parser.add_argument('--task_name', type=str, default='classification',
                        help='任务类型：classification')
    parser.add_argument('--num_class', type=int, default=2,
                        help='分类类别数（DEAP:2, SEED:3）')
    
    # 数据配置
    parser.add_argument('--data', type=str, default='DEAP', 
                        choices=['DEAP', 'SEED'], help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./dataset/DEAP/',
                        help='数据集根目录')
    parser.add_argument('--data_path', type=str, default='data.csv',
                        help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='特征类型')
    parser.add_argument('--target', type=str, default='label',
                        help='目标列名')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='checkpoint保存位置')
    
    # 数据维度配置
    parser.add_argument('--seq_len', type=int, default=256,
                        help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=0,
                        help='标签长度（分类任务为0）')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='预测长度（分类任务为0）')
    
    # 模型维度配置
    parser.add_argument('--enc_in', type=int, default=32,
                        help='输入通道数（DEAP:32, SEED:62）')
    parser.add_argument('--dec_in', type=int, default=32,
                        help='解码器输入维度')
    parser.add_argument('--c_out', type=int, default=32,
                        help='输出维度')
    parser.add_argument('--d_model', type=int, default=32,
                        help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=64,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间编码方式')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间频率')
    parser.add_argument('--factor', type=int, default=1,
                        help='注意力因子')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='激活函数')
    parser.add_argument('--output_attention', action='store_true',
                        help='是否输出注意力')
    parser.add_argument('--distil', action='store_true', default=True,
                        help='是否使用蒸馏')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='移动平均窗口')
    
    # 损失函数配置（新增）
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'weighted_ce', 'focal', 'label_smoothing'],
                        help='损失函数类型')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑系数')
    
    # 优化器配置（新增）
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential'],
                        help='学习率调度器类型')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='预热轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    
    # 训练配置
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--lradj', type=str, default='COS',
                        help='学习率调整策略')
    parser.add_argument('--pct_start', type=float, default=0.2,
                        help='OneCycleLR的pct_start')
    parser.add_argument('--use_amp', action='store_true',
                        help='是否使用混合精度训练')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='评估批量大小')
    parser.add_argument('--des', type=str, default='enhanced',
                        help='实验描述')
    parser.add_argument('--itr', type=int, default=1,
                        help='实验重复次数')
    
    # LLM配置
    parser.add_argument('--llm_model', type=str, default='GPT2',
                        help='LLM模型类型')
    parser.add_argument('--llm_dim', type=int, default=768,
                        help='LLM模型维度')
    parser.add_argument('--llm_layers', type=int, default=6,
                        help='使用的LLM层数')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='补丁长度')
    parser.add_argument('--stride', type=int, default=8,
                        help='步长')
    parser.add_argument('--prompt_domain', type=int, default=0,
                        help='提示域')
    parser.add_argument('--temperature', type=int, default=10,
                        help='温度参数')
    parser.add_argument('--model_comment', type=str, default='enhanced',
                        help='模型注释')
    parser.add_argument('--percent', type=int, default=100,
                        help='数据使用百分比')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='季节性模式')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='预测任务损失函数')
    parser.add_argument('--align_epochs', type=int, default=10,
                        help='对齐轮数')
    
    return parser


def prepare_colab_env():
    """准备Colab环境"""
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("检测到Google Colab环境")
        # 安装必要的包
        os.system('pip install accelerate transformers -q')
        # 挂载Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
    else:
        print("本地环境")
    
    return IN_COLAB


def setup_paths(args, IN_COLAB):
    """设置路径"""
    if IN_COLAB:
        # 固定的数据集路径
        if args.data == 'DEAP':
            args.root_path = '/content/drive/MyDrive/EEG_Data/DEAP/'
        elif args.data == 'SEED':
            args.root_path = '/content/drive/MyDrive/EEG_Data/SEED/'
        else:
            # 尝试自动检测
            possible_paths = [
                f'/content/drive/MyDrive/EEG_Data/{args.data}/',
                f'/content/drive/MyDrive/{args.data}/',
                f'/content/drive/MyDrive/dataset/{args.data}/'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    args.root_path = path
                    break
            else:
                args.root_path = f'/content/drive/MyDrive/EEG_Data/{args.data}/'
        
        # 检查点路径
        args.checkpoints = '/content/drive/MyDrive/Time-LLM-EEG/checkpoints/'
        os.makedirs(args.checkpoints, exist_ok=True)
    
    # 创建必要的目录
    os.makedirs(args.checkpoints, exist_ok=True)
    
    print(f"配置路径:")
    print(f"  - 数据路径: {args.root_path}")
    print(f"  - 检查点路径: {args.checkpoints}")


def print_args(args):
    """打印所有参数"""
    print("\n" + "=" * 70)
    print("增强版训练参数配置")
    print("=" * 70)
    
    # 按类别打印参数
    print("\n[任务配置]")
    print(f"  - task_name: {args.task_name}")
    print(f"  - num_class: {args.num_class}")
    print(f"  - model: {args.model}")
    
    print("\n[数据配置]")
    print(f"  - data: {args.data}")
    print(f"  - seq_len: {args.seq_len}")
    print(f"  - enc_in: {args.enc_in}")
    
    print("\n[损失函数配置]")
    print(f"  - loss_type: {args.loss_type}")
    if args.loss_type == 'focal':
        print(f"  - focal_alpha: {args.focal_alpha}")
        print(f"  - focal_gamma: {args.focal_gamma}")
    elif args.loss_type == 'label_smoothing':
        print(f"  - label_smoothing: {args.label_smoothing}")
    
    print("\n[优化器配置]")
    print(f"  - optimizer: {args.optimizer}")
    print(f"  - scheduler_type: {args.scheduler_type}")
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - weight_decay: {args.weight_decay}")
    print(f"  - warmup_epochs: {args.warmup_epochs}")
    print(f"  - grad_clip: {args.grad_clip}")
    
    print("\n[模型配置]")
    print(f"  - d_model: {args.d_model}")
    print(f"  - d_ff: {args.d_ff}")
    print(f"  - patch_len: {args.patch_len}")
    print(f"  - stride: {args.stride}")
    
    print("\n[LLM配置]")
    print(f"  - llm_model: {args.llm_model}")
    print(f"  - llm_layers: {args.llm_layers}")
    
    print("\n[训练配置]")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - train_epochs: {args.train_epochs}")
    print(f"  - patience: {args.patience}")


def run_training(args):
    """运行训练"""
    print("\n" + "=" * 70)
    print("开始增强版训练")
    print("=" * 70)
    
    # 构建命令 - 使用增强版run_main
    cmd = f"python run_main_enhanced.py"
    
    # 添加所有参数
    for key, value in vars(args).items():
        if value is not None and value != '' and value != False:
            if isinstance(value, bool) and value:
                cmd += f" --{key}"
            else:
                cmd += f" --{key} {value}"
    
    print(f"\n执行命令:")
    print(cmd)
    print("\n" + "-" * 70)
    
    # 执行训练
    os.system(cmd)


def main():
    """主函数"""
    # 创建参数解析器
    parser = create_parser()
    args = parser.parse_args()
    
    # 准备环境
    IN_COLAB = prepare_colab_env()
    
    # 设置路径
    setup_paths(args, IN_COLAB)
    
    # 根据数据集调整参数
    if args.data == 'DEAP':
        args.enc_in = 32
        args.dec_in = 32
        args.c_out = 32
        args.num_class = 2  # 二分类
        # DEAP推荐使用Focal Loss
        if args.loss_type == 'focal':
            args.focal_alpha = 0.25
            args.focal_gamma = 2.0
    elif args.data == 'SEED':
        args.enc_in = 62
        args.dec_in = 62
        args.c_out = 62
        args.num_class = 3  # 三分类
        # SEED推荐使用加权CE
        if args.loss_type == 'weighted_ce':
            pass  # 会自动计算权重
    
    # 生成实验ID
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.model_id = f"{args.model}_{args.data}_{args.loss_type}_{current_time}"
    
    # 打印参数
    print_args(args)
    
    # 确认是否开始训练
    response = input("\n是否开始训练? (y/n): ")
    if response.lower() == 'y':
        run_training(args)
    else:
        print("训练已取消")


if __name__ == '__main__':
    main()


# ============================================================
# 使用示例（在Colab中）
# ============================================================
"""
# 1. 基础训练（DEAP数据集，Focal Loss）
!python train_eeg_enhanced.py --data DEAP --loss_type focal

# 2. SEED数据集，加权交叉熵
!python train_eeg_enhanced.py --data SEED --loss_type weighted_ce --batch_size 8

# 3. 使用标签平滑
!python train_eeg_enhanced.py --data DEAP --loss_type label_smoothing --label_smoothing 0.1

# 4. 自定义优化器和调度器
!python train_eeg_enhanced.py --optimizer adamw --scheduler_type cosine --warmup_epochs 10

# 5. 快速测试（少量epoch）
!python train_eeg_enhanced.py --train_epochs 5 --batch_size 4

# 6. 使用混合精度训练（加速）
!python train_eeg_enhanced.py --use_amp

# 7. 调整模型大小（针对显存）
!python train_eeg_enhanced.py --d_model 16 --d_ff 32 --batch_size 8
"""
