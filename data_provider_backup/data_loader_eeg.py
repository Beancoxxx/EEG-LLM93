"""
EEG数据加载器
支持DEAP和SEED数据集的加载
包含数据预处理、滑动窗口、标准化等功能
"""

import os
import pickle
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from torch.utils.data import Dataset
import torch
import warnings

warnings.filterwarnings('ignore')


class Dataset_EEG_Base(Dataset):
    """EEG数据集基类"""

    def __init__(self, seq_len=256, pred_len=0, label_len=0):
        """
        参数:
            seq_len: 输入序列长度（默认256个时间点）
            pred_len: 预测长度（分类任务设为0）
            label_len: 标签长度（分类任务设为0）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Dataset_DEAP(Dataset_EEG_Base):
    """DEAP数据集加载器

    数据结构:
    - 原始数据: (40 trials, 40 channels, 8064 time_points)
    - 前32个通道是EEG数据
    - 标签: valence, arousal, dominance, liking (1-9评分)
    - 采样率: 128 Hz
    """

    def __init__(self, root_path, flag='train', seq_len=256, pred_len=0, label_len=0,
                 n_class=2, classification_type='valence', subject_list=None,
                 overlap=0, normalize=True, filter_freq=None, sampling_rate=128):
        """
        参数:
            root_path: DEAP数据集根目录
            flag: 'train'/'val'/'test'
            seq_len: 序列长度
            n_class: 分类数量（2或4）
            classification_type: 'valence'或'arousal'（仅用于二分类）
            subject_list: 要使用的被试列表，如['s01', 's02']
            overlap: 滑动窗口重叠量（0-seq_len之间）
            normalize: 是否标准化数据
            filter_freq: 滤波频段，如(0.5, 45)表示0.5-45Hz带通滤波
            sampling_rate: 采样率（DEAP为128Hz）
        """
        super().__init__(seq_len, pred_len, label_len)

        self.root_path = root_path
        self.flag = flag
        self.n_class = n_class
        self.classification_type = classification_type
        self.overlap = overlap
        self.normalize = normalize
        self.filter_freq = filter_freq
        self.sampling_rate = sampling_rate

        # 验证overlap参数
        if overlap < 0 or overlap >= seq_len:
            raise ValueError(f"overlap必须在0到{seq_len - 1}之间，当前值：{overlap}")

        # 如果没有指定被试列表，使用默认分配
        if subject_list is None:
            # 默认: s01-s20 训练, s21-s26 验证, s27-s32 测试
            if flag == 'train':
                subject_list = [f's{i:02d}' for i in range(1, 21)]
            elif flag == 'val':
                subject_list = [f's{i:02d}' for i in range(21, 27)]
            else:  # test
                subject_list = [f's{i:02d}' for i in range(27, 33)]

        self.subject_list = subject_list

        # 加载数据
        self._load_data()

    def _bandpass_filter(self, data, low_freq, high_freq, fs):
        """带通滤波

        参数:
            data: (n_channels, n_samples)
            low_freq: 低频截止频率
            high_freq: 高频截止频率
            fs: 采样率
        """
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # 设计巴特沃斯滤波器
        b, a = signal.butter(4, [low, high], btype='band')

        # 对每个通道进行滤波
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])

        return filtered_data

    def _normalize_trial(self, data):
        """标准化单个试验的数据

        参数:
            data: (n_channels, n_samples)
        返回:
            标准化后的数据
        """
        # 对每个通道独立进行标准化
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        # 避免除零
        std[std == 0] = 1
        return (data - mean) / std

    def _load_data(self):
        """加载所有被试的数据"""
        all_data = []
        all_labels = []

        print(f"正在加载DEAP {self.flag} 数据集...")

        for subject in self.subject_list:
            file_path = os.path.join(self.root_path, f'{subject}.dat')
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 {file_path}")
                continue

            # 读取数据
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')

            # 提取EEG数据 (只要前32个通道)
            eeg_data = data_dict['data'][:, :32, :]  # (40, 32, 8064)
            labels = data_dict['labels']  # (40, 4)

            # 处理每个试验
            for trial_idx in range(eeg_data.shape[0]):
                trial_data = eeg_data[trial_idx]  # (32, 8064)
                trial_label = self._process_label(labels[trial_idx])

                # 数据预处理
                # 1. 滤波（如果指定）
                if self.filter_freq is not None:
                    trial_data = self._bandpass_filter(
                        trial_data,
                        self.filter_freq[0],
                        self.filter_freq[1],
                        self.sampling_rate
                    )

                # 2. 标准化（如果指定）
                if self.normalize:
                    trial_data = self._normalize_trial(trial_data)

                # 3. 滑动窗口分割
                if self.overlap > 0:
                    # 使用滑动窗口
                    step = self.seq_len - self.overlap
                    n_segments = (trial_data.shape[1] - self.seq_len) // step + 1

                    for seg_idx in range(n_segments):
                        start = seg_idx * step
                        end = start + self.seq_len

                        if end <= trial_data.shape[1]:
                            segment = trial_data[:, start:end]  # (32, seq_len)
                            all_data.append(segment)
                            all_labels.append(trial_label)
                else:
                    # 无重叠分割
                    n_segments = trial_data.shape[1] // self.seq_len

                    for seg_idx in range(n_segments):
                        start = seg_idx * self.seq_len
                        end = start + self.seq_len

                        segment = trial_data[:, start:end]  # (32, seq_len)
                        all_data.append(segment)
                        all_labels.append(trial_label)

        # 转换为numpy数组
        self.data = np.array(all_data)  # (n_samples, 32, seq_len)
        self.labels = np.array(all_labels)  # (n_samples,)

        # 打印数据集信息
        print(f"DEAP {self.flag} 数据集加载完成:")
        print(f"  - 样本数: {len(self.data)}")
        print(f"  - 数据形状: {self.data.shape}")
        print(f"  - 标签分布: {np.bincount(self.labels)}")
        if self.overlap > 0:
            print(f"  - 滑动窗口重叠: {self.overlap} 个时间点")
        if self.filter_freq is not None:
            print(f"  - 滤波范围: {self.filter_freq[0]}-{self.filter_freq[1]} Hz")
        if self.normalize:
            print(f"  - 数据已标准化")

    def _process_label(self, label_vec):
        """处理标签
        label_vec: [valence, arousal, dominance, liking]
        """
        valence, arousal = label_vec[0], label_vec[1]

        if self.n_class == 2:
            # 二分类
            if self.classification_type == 'valence':
                return 0 if valence < 5 else 1
            else:  # arousal
                return 0 if arousal < 5 else 1
        else:  # 4分类
            # 基于valence和arousal的四象限
            if valence >= 5 and arousal >= 5:
                return 0  # 高兴 (HVHA)
            elif valence < 5 and arousal >= 5:
                return 1  # 愤怒 (LVHA)
            elif valence < 5 and arousal < 5:
                return 2  # 悲伤 (LVLA)
            else:
                return 3  # 平静 (HVLA)

    def __getitem__(self, index):
        """
        返回:
            x: (seq_len, n_channels) - 转置以匹配Time-LLM的输入格式
            y: 标签
            x_mark: 时间特征（这里用零填充）
            y_mark: 时间特征（这里用零填充）
        """
        x = self.data[index]  # (32, seq_len)
        y = self.labels[index]

        # 转置以匹配Time-LLM的期望输入 (seq_len, n_channels)
        x = x.T  # (seq_len, 32)

        # 时间特征（分类任务中可以用零填充）
        x_mark = np.zeros((self.seq_len, 4))
        y_mark = np.zeros((self.pred_len + self.label_len, 4))

        return x, y, x_mark, y_mark

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        """逆变换（分类任务不需要）"""
        return data


class Dataset_SEED(Dataset_EEG_Base):
    """SEED数据集加载器

    数据结构:
    - 每个文件15个试验
    - 每个试验: (62 channels, variable time_points)
    - 标签: -1(负面), 0(中性), 1(正面)
    - 采样率: 200 Hz
    """

    def __init__(self, root_path, flag='train', seq_len=256, pred_len=0, label_len=0,
                 n_class=3, subject_list=None, overlap=0, normalize=True,
                 filter_freq=None, sampling_rate=200):
        """
        参数:
            root_path: SEED数据集根目录
            flag: 'train'/'val'/'test'
            seq_len: 序列长度
            n_class: 分类数量（2或3）
            subject_list: 要使用的被试文件列表
            overlap: 滑动窗口重叠量
            normalize: 是否标准化数据
            filter_freq: 滤波频段，如(0.5, 45)
            sampling_rate: 采样率（SEED为200Hz）
        """
        super().__init__(seq_len, pred_len, label_len)

        self.root_path = root_path
        self.flag = flag
        self.n_class = n_class
        self.overlap = overlap
        self.normalize = normalize
        self.filter_freq = filter_freq
        self.sampling_rate = sampling_rate

        # 验证overlap参数
        if overlap < 0 or overlap >= seq_len:
            raise ValueError(f"overlap必须在0到{seq_len - 1}之间，当前值：{overlap}")

        # SEED的标签顺序（15个试验）
        self.trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        # 获取所有mat文件
        all_files = [f for f in os.listdir(root_path) if f.endswith('.mat') and f != 'label.mat']
        all_files.sort()

        # 如果没有指定文件列表，使用默认分配
        if subject_list is None:
            # 假设有15个被试，按3:1:1分配
            n_files = len(all_files)
            n_train = int(n_files * 0.6)
            n_val = int(n_files * 0.2)

            if flag == 'train':
                subject_list = all_files[:n_train]
            elif flag == 'val':
                subject_list = all_files[n_train:n_train + n_val]
            else:  # test
                subject_list = all_files[n_train + n_val:]

        self.subject_list = subject_list

        # 加载数据
        self._load_data()

    def _bandpass_filter(self, data, low_freq, high_freq, fs):
        """带通滤波（与DEAP相同）"""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = signal.butter(4, [low, high], btype='band')

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])

        return filtered_data

    def _normalize_trial(self, data):
        """标准化单个试验的数据（与DEAP相同）"""
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std

    def _load_data(self):
        """加载所有被试的数据"""
        all_data = []
        all_labels = []

        print(f"正在加载SEED {self.flag} 数据集...")

        for file_name in self.subject_list:
            file_path = os.path.join(self.root_path, file_name)
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 {file_path}")
                continue

            # 读取mat文件
            mat_data = sio.loadmat(file_path)

            # 获取所有的EEG试验键（排除matlab元数据和其他非EEG数据）
            eeg_keys = []
            for key in mat_data.keys():
                if not key.startswith('__') and 'eeg' in key.lower():
                    eeg_keys.append(key)

            # 按照试验编号排序（例如：xxx_eeg1, xxx_eeg2, ...）
            eeg_keys.sort(key=lambda x: int(x.split('eeg')[-1]))

            # 检查试验数量
            if len(eeg_keys) != 15:
                print(f"警告: {file_name} 中找到 {len(eeg_keys)} 个试验，期望15个")
                print(f"找到的试验: {eeg_keys}")

            # 处理每个试验
            for trial_idx, trial_name in enumerate(eeg_keys[:15]):  # 最多处理15个
                trial_data = mat_data[trial_name]  # (62, time_points)
                original_label = self.trial_labels[trial_idx]

                # 处理标签
                if self.n_class == 2:
                    # 二分类：只保留正面和负面，跳过中性
                    if original_label == 0:  # 中性
                        continue
                    trial_label = 1 if original_label == 1 else 0
                else:  # 3分类
                    # 将-1, 0, 1映射到0, 1, 2
                    trial_label = original_label + 1

                # 数据预处理
                # 1. 滤波（如果指定）
                if self.filter_freq is not None:
                    trial_data = self._bandpass_filter(
                        trial_data,
                        self.filter_freq[0],
                        self.filter_freq[1],
                        self.sampling_rate
                    )

                # 2. 标准化（如果指定）
                if self.normalize:
                    trial_data = self._normalize_trial(trial_data)

                # 3. 滑动窗口分割
                if self.overlap > 0:
                    # 使用滑动窗口
                    step = self.seq_len - self.overlap
                    n_segments = (trial_data.shape[1] - self.seq_len) // step + 1

                    for seg_idx in range(n_segments):
                        start = seg_idx * step
                        end = start + self.seq_len

                        if end <= trial_data.shape[1]:
                            segment = trial_data[:, start:end]  # (62, seq_len)
                            all_data.append(segment)
                            all_labels.append(trial_label)
                else:
                    # 无重叠分割
                    n_segments = trial_data.shape[1] // self.seq_len

                    for seg_idx in range(n_segments):
                        start = seg_idx * self.seq_len
                        end = start + self.seq_len

                        segment = trial_data[:, start:end]  # (62, seq_len)
                        all_data.append(segment)
                        all_labels.append(trial_label)

        # 转换为numpy数组
        self.data = np.array(all_data)  # (n_samples, 62, seq_len)
        self.labels = np.array(all_labels)  # (n_samples,)

        # 打印数据集信息
        print(f"SEED {self.flag} 数据集加载完成:")
        print(f"  - 样本数: {len(self.data)}")
        print(f"  - 数据形状: {self.data.shape}")
        if self.n_class == 3:
            # 3分类情况
            label_counts = np.bincount(self.labels)
            print(f"  - 标签分布: 负面={label_counts[0]}, 中性={label_counts[1]}, 正面={label_counts[2]}")
        else:
            # 2分类情况
            label_counts = np.bincount(self.labels)
            print(f"  - 标签分布: 负面={label_counts[0]}, 正面={label_counts[1]}")

        if self.overlap > 0:
            print(f"  - 滑动窗口重叠: {self.overlap} 个时间点")
        if self.filter_freq is not None:
            print(f"  - 滤波范围: {self.filter_freq[0]}-{self.filter_freq[1]} Hz")
        if self.normalize:
            print(f"  - 数据已标准化")

    def __getitem__(self, index):
        """
        返回:
            x: (seq_len, n_channels) - 转置以匹配Time-LLM的输入格式
            y: 标签
            x_mark: 时间特征（这里用零填充）
            y_mark: 时间特征（这里用零填充）
        """
        x = self.data[index]  # (62, seq_len)
        y = self.labels[index]

        # 转置以匹配Time-LLM的期望输入 (seq_len, n_channels)
        x = x.T  # (seq_len, 62)

        # 时间特征（分类任务中可以用零填充）
        x_mark = np.zeros((self.seq_len, 4))
        y_mark = np.zeros((self.pred_len + self.label_len, 4))

        return x, y, x_mark, y_mark

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        """逆变换（分类任务不需要）"""
        return data


# 测试函数
if __name__ == "__main__":
    print("=" * 70)
    print("测试改进后的EEG数据加载器")
    print("=" * 70)

    # 测试DEAP数据集
    print("\n1. 测试DEAP基础功能...")
    deap_dataset_basic = Dataset_DEAP(
        root_path=r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python",
        flag='train',
        seq_len=256,
        n_class=2,
        classification_type='valence',
        subject_list=['s01'],  # 只用一个被试测试
        overlap=0,
        normalize=False,
        filter_freq=None
    )

    print("\n2. 测试DEAP滑动窗口功能...")
    deap_dataset_overlap = Dataset_DEAP(
        root_path=r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python",
        flag='train',
        seq_len=256,
        n_class=2,
        classification_type='valence',
        subject_list=['s01'],
        overlap=128,  # 50%重叠
        normalize=True,
        filter_freq=None
    )

    print("\n3. 测试DEAP滤波功能...")
    deap_dataset_filter = Dataset_DEAP(
        root_path=r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python",
        flag='train',
        seq_len=256,
        n_class=2,
        classification_type='valence',
        subject_list=['s01'],
        overlap=0,
        normalize=True,
        filter_freq=(0.5, 45)  # 0.5-45 Hz带通滤波
    )

    # 获取一个样本
    x, y, x_mark, y_mark = deap_dataset_filter[0]
    print(f"\n样本形状:")
    print(f"  - x: {x.shape}")
    print(f"  - y: {y}")
    print(f"  - x数据范围: [{x.min():.4f}, {x.max():.4f}]")

    # 测试SEED数据集
    print("\n" + "=" * 70)
    print("\n4. 测试SEED二分类（过滤中性）...")
    seed_dataset_binary = Dataset_SEED(
        root_path=r"D:\文件\文件\HKU\Dissertation\dataset\SEED\SEED\Preprocessed_EEG",
        flag='train',
        seq_len=256,
        n_class=2,  # 二分类，会自动过滤中性样本
        subject_list=['1_20131027.mat'],
        overlap=128,
        normalize=True
    )

    print("\n5. 测试SEED三分类...")
    seed_dataset_ternary = Dataset_SEED(
        root_path=r"D:\文件\文件\HKU\Dissertation\dataset\SEED\SEED\Preprocessed_EEG",
        flag='train',
        seq_len=256,
        n_class=3,  # 三分类
        subject_list=['1_20131027.mat'],
        overlap=0,
        normalize=True,
        filter_freq=(0.5, 50)  # SEED采样率更高，可以保留更高频率
    )

    # 获取一个样本
    x, y, x_mark, y_mark = seed_dataset_ternary[0]
    print(f"\n样本形状:")
    print(f"  - x: {x.shape}")
    print(f"  - y: {y}")
    print(f"  - x数据范围: [{x.min():.4f}, {x.max():.4f}]")

    print("\n" + "=" * 70)
    print("测试完成！")