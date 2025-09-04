"""
EEG情绪分类训练脚本
适用于在Google Colab中训练TimeLLM进行EEG情绪分类
"""

import os
import sys
import argparse
from datetime import datetime

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='TimeLLM for EEG Emotion Classification')
    
    # 基本配置
    parser.add_argument('--is_training', type=int, default=1, help='是否训练')
    parser.add_argument('--model_id', type=str, default='test', help='模型ID')
    parser.add_argument('--model', type=str, default='TimeLLM', help='模型名称')
    
    # 任务配置 - 关键修改
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
                        help='数据文件名（EEG数据集使用默认值）')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务的特征类型（分类任务保持M）')
    parser.add_argument('--target', type=str, default='label',
                        help='目标列名（分类任务）')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='checkpoint保存位置')
    
    # 数据维度配置
    parser.add_argument('--seq_len', type=int, default=256,
                        help='输入序列长度（EEG片段长度）')
    parser.add_argument('--label_len', type=int, default=0,
                        help='标签长度（分类任务为0）')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='预测长度（分类任务为0）')
    
    # 模型维度配置
    parser.add_argument('--enc_in', type=int, default=32,
                        help='输入通道数（DEAP:32, SEED:62）')
    parser.add_argument('--dec_in', type=int, default=32,
                        help='解码器输入维度（分类任务等于enc_in）')
    parser.add_argument('--c_out', type=int, default=32,
                        help='输出维度（分类任务等于enc_in）')
    parser.add_argument('--d_model', type=int, default=32,
                        help='模型维度')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=64,
                        help='FFN维度')
    parser.add_argument('--factor', type=int, default=1,
                        help='衰减因子')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout比率')
    
    # LLM配置 - 针对Colab优化
    parser.add_argument('--llm_model', type=str, default='GPT2',
                        help='LLM模型（GPT2/LLAMA）')
    parser.add_argument('--llm_dim', type=int, default=768,
                        help='LLM维度（GPT2:768, LLAMA:4096）')
    parser.add_argument('--llm_layers', type=int, default=2,
                        help='使用的LLM层数（减少内存使用）')
    
    # Patch配置
    parser.add_argument('--patch_len', type=int, default=16,
                        help='patch长度')
    parser.add_argument('--stride', type=int, default=8,
                        help='patch步长')
    
    # 训练配置
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')
    parser.add_argument('--itr', type=int, default=1,
                        help='实验重复次数')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--des', type=str, default='Exp',
                        help='实验描述')
    parser.add_argument('--loss', type=str, default='CrossEntropy',
                        help='损失函数（分类任务使用CrossEntropy）')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='是否使用混合精度训练')
    
    # 其他配置
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='激活函数')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征频率')
    parser.add_argument('--percent', type=int, default=100,
                        help='数据使用百分比')
    
    return parser


def prepare_colab_env():
    """准备Colab环境"""
    print("=" * 70)
    print("准备Colab环境")
    print("=" * 70)
    
    # 检查是否在Colab环境
    try:
        import google.colab
        IN_COLAB = True
        print("✓ 检测到Colab环境")
    except:
        IN_COLAB = False
        print("✗ 非Colab环境")
    
    # 检查Google Drive是否已经挂载
    if IN_COLAB and os.path.exists('/content/drive'):
        print("✓ Google Drive已经挂载")
    elif IN_COLAB:
        try:
            # 尝试在notebook环境中挂载
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive已挂载")
        except:
            print("⚠ 无法自动挂载Google Drive")
            print("  请在notebook中手动运行: from google.colab import drive; drive.mount('/content/drive')")
    
    return IN_COLAB


def setup_paths(args, IN_COLAB):
    """设置路径"""
    if IN_COLAB:
        # Colab路径配置
        project_path = "/content/timellm/Time-LLM-main"
        
        # 检查项目是否存在，如果不存在则克隆
        if not os.path.exists(project_path):
            print("正在克隆Time-LLM项目...")
            os.system("git clone https://github.com/KimMeen/Time-LLM.git")
        
        # 添加项目路径
        sys.path.append(project_path)
        os.chdir(project_path)
        
        # 设置数据路径（从Google Drive）
        if args.data == 'DEAP':
            args.root_path = '/content/drive/MyDrive/EEG_Data/DEAP/data_preprocessed_python'
        elif args.data == 'SEED':
            args.root_path = '/content/drive/MyDrive/EEG_Data/SEED/SEED/Preprocessed_EEG'
    
    print(f"\n路径配置:")
    print(f"  - 项目路径: {os.getcwd()}")
    print(f"  - 数据路径: {args.root_path}")
    print(f"  - 检查点路径: {args.checkpoints}")


def print_args(args):
    """打印所有参数"""
    print("\n" + "=" * 70)
    print("训练参数配置")
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
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - train_epochs: {args.train_epochs}")
    

def run_training(args):
    """运行训练"""
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    # 构建命令 - 使用修改版的run_main
    cmd = f"python run_main_modified.py"
    
    # 添加所有参数（过滤空值和False值）
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
    elif args.data == 'SEED':
        args.enc_in = 62
        args.dec_in = 62
        args.c_out = 62
        args.num_class = 3  # 三分类
    
    # 生成实验ID
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.model_id = f"{args.model}_{args.data}_{args.task_name}_{current_time}"
    
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
# 1. 基础训练（DEAP数据集，默认参数）
!python train_eeg_classification.py

# 2. 自定义参数训练
!python train_eeg_classification.py --data DEAP --batch_size 8 --train_epochs 5 --learning_rate 0.0001

# 3. SEED数据集训练
!python train_eeg_classification.py --data SEED --num_class 3 --enc_in 62

# 4. 快速测试（少量epoch）
!python train_eeg_classification.py --train_epochs 2 --batch_size 4

# 5. 使用LLAMA模型（需要更多显存）
!python train_eeg_classification.py --llm_model LLAMA --llm_dim 4096 --llm_layers 1
"""