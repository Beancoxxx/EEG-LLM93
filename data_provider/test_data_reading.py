"""
检查DEAP和SEED数据集的读取
这个文件用来验证我们能否正确读取两种格式的数据
"""

import pickle
import scipy.io as sio
import numpy as np
import os

def check_deap_reading(file_path):
    """检查读取DEAP数据集
    
    DEAP数据说明：
    - 使用pickle格式存储
    - 每个文件包含一个字典，有'data'和'labels'两个key
    - data形状：(40, 40, 8064) = (trials, channels, time_points)
      其中前32个通道是EEG数据
    - labels形状：(40, 4) = (trials, label_types)
      4种标签：valence, arousal, dominance, liking
    """
    print("="*50)
    print("检查DEAP数据读取")
    print("="*50)
    
    try:
        # 读取pickle文件
        print(f"正在读取: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print(f"✓ 成功读取文件!")
        print(f"数据类型: {type(data)}")
        print(f"包含的keys: {list(data.keys())}")
        
        # 查看数据形状
        eeg_data = data['data']
        labels = data['labels']
        
        print(f"\nEEG数据形状: {eeg_data.shape}")
        print(f"  - 试验数(trials): {eeg_data.shape[0]}")
        print(f"  - 通道数(channels): {eeg_data.shape[1]}")
        print(f"  - 时间点数(time_points): {eeg_data.shape[2]}")
        
        print(f"\n标签形状: {labels.shape}")
        print(f"  - 标签类型: valence, arousal, dominance, liking")
        print(f"  - 第一个试验的标签: {labels[0]}")
        
        # 只取EEG通道（前32个）
        eeg_only = eeg_data[:, :32, :]
        print(f"\n只取EEG通道后的形状: {eeg_only.shape}")
        
        # 显示一些统计信息
        print(f"\nEEG数据统计:")
        print(f"  - 最小值: {eeg_only.min():.4f}")
        print(f"  - 最大值: {eeg_only.max():.4f}")
        print(f"  - 平均值: {eeg_only.mean():.4f}")
        print(f"  - 标准差: {eeg_only.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 读取DEAP数据失败: {e}")
        print("请确保文件路径正确，且文件是.dat格式")
        return False

def check_seed_reading(file_path):
    """检查读取SEED数据集
    
    SEED数据说明：
    - 使用.mat格式存储
    - 每个文件包含15个试验（djc_eeg1到djc_eeg15）
    - 每个试验的数据形状：(62, n_samples) = (channels, time_points)
    - 标签文件单独存储，15个试验的标签为：[1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
    """
    print("\n" + "="*50)
    print("检查SEED数据读取")
    print("="*50)
    
    try:
        # 读取mat文件
        print(f"正在读取: {file_path}")
        mat_data = sio.loadmat(file_path)
        
        print(f"✓ 成功读取文件!")
        
        # 获取所有的试验名称（排除matlab的元数据）
        trial_names = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"文件中的试验: {trial_names}")
        
        # 查看每个试验的数据
        for i, trial_name in enumerate(trial_names[:3]):  # 只显示前3个
            if trial_name in mat_data:
                trial_data = mat_data[trial_name]
                print(f"\n试验{i+1} ({trial_name})的数据形状: {trial_data.shape}")
                if i == 0:  # 只对第一个试验显示详细信息
                    print(f"  - 通道数(channels): {trial_data.shape[0]}")
                    print(f"  - 时间点数(time_points): {trial_data.shape[1]}")
                    
                    # 显示统计信息
                    print(f"\nEEG数据统计:")
                    print(f"  - 最小值: {trial_data.min():.4f}")
                    print(f"  - 最大值: {trial_data.max():.4f}")
                    print(f"  - 平均值: {trial_data.mean():.4f}")
                    print(f"  - 标准差: {trial_data.std():.4f}")
        
        # 读取标签文件
        label_path = os.path.join(os.path.dirname(file_path), 'label.mat')
        if os.path.exists(label_path):
            print(f"\n正在读取标签文件...")
            label_data = sio.loadmat(label_path)
            labels = label_data['label'][0]
            print(f"✓ 标签数据: {labels}")
            print(f"标签说明: 1=正面(positive), 0=中性(neutral), -1=负面(negative)")
        else:
            print(f"\n✗ 标签文件不存在: {label_path}")
            
        return True
        
    except Exception as e:
        print(f"✗ 读取SEED数据失败: {e}")
        print("请确保文件路径正确，且文件是.mat格式")
        return False

def main():
    """主函数：检查两种数据集的读取"""
    
    print("开始检查EEG数据集读取...\n")
    
    # 使用你提供的路径
    deap_path = r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python\s01.dat"
    seed_path = r"D:\文件\文件\HKU\Dissertation\dataset\SEED\SEED\Preprocessed_EEG\1_20131027.mat"
    
    # 检查DEAP
    if os.path.exists(deap_path):
        success_deap = check_deap_reading(deap_path)
    else:
        print(f"✗ DEAP文件不存在: {deap_path}")
        print("请检查路径是否正确")
        success_deap = False
    
    # 检查SEED  
    if os.path.exists(seed_path):
        success_seed = check_seed_reading(seed_path)
    else:
        print(f"\n✗ SEED文件不存在: {seed_path}")
        print("请检查路径是否正确")
        success_seed = False
    
    # 总结
    print("\n" + "="*50)
    print("检查总结")
    print("="*50)
    print(f"DEAP数据读取: {'✓ 成功' if success_deap else '✗ 失败'}")
    print(f"SEED数据读取: {'✓ 成功' if success_seed else '✗ 失败'}")
    
    if success_deap and success_seed:
        print("\n太棒了！两个数据集都能正确读取。")
        print("下一步我们可以开始创建数据加载器了。")
    else:
        print("\n请先解决上面的问题，确保数据能够正确读取。")

if __name__ == "__main__":
    main()
