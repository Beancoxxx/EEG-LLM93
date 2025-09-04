"""
修改后的data_factory.py
在原有基础上添加EEG数据集（DEAP和SEED）的支持
"""

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
#from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

# 导入EEG数据加载器
from data_provider.data_loader_eeg import Dataset_DEAP, Dataset_SEED

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    #'PSM': PSMSegLoader,
    #'MSL': MSLSegLoader,
    #'SMAP': SMAPSegLoader,
    #'SMD': SMDSegLoader,
    #'SWAT': SWATSegLoader,
    #'UEA': UEAloader,
    # 添加EEG数据集
    'DEAP': Dataset_DEAP,
    'SEED': Dataset_SEED,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # 处理EEG数据集
    if args.data in ['DEAP', 'SEED']:
        # EEG数据集的特殊参数处理
        if flag == 'test':
            shuffle_flag = False
            drop_last = False  # 测试集不丢弃最后一批
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        # 创建EEG数据集实例
        if args.data == 'DEAP':
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
                seq_len=args.seq_len,
                pred_len=args.pred_len if hasattr(args, 'pred_len') else 0,
                label_len=args.label_len if hasattr(args, 'label_len') else 0,
                n_class=args.num_class,
                classification_type=getattr(args, 'classification_type', 'valence'),
                subject_list=getattr(args, 'subject_list', None),
                overlap=getattr(args, 'overlap', 0),
                normalize=getattr(args, 'normalize', True),
                filter_freq=getattr(args, 'filter_freq', None),
                sampling_rate=getattr(args, 'sampling_rate', 128)
            )
        else:  # SEED
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
                seq_len=args.seq_len,
                pred_len=args.pred_len if hasattr(args, 'pred_len') else 0,
                label_len=args.label_len if hasattr(args, 'label_len') else 0,
                n_class=args.num_class,
                subject_list=getattr(args, 'subject_list', None),
                overlap=getattr(args, 'overlap', 0),
                normalize=getattr(args, 'normalize', True),
                filter_freq=getattr(args, 'filter_freq', None),
                sampling_rate=getattr(args, 'sampling_rate', 200)
            )

        # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    # 原有数据集的处理逻辑
    elif args.data == 'm4':
        drop_last = False
        if flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    elif args.data == 'UEA':
        drop_last = False
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
        else:
            shuffle_flag = True
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader

    else:
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader


# 测试代码
if __name__ == "__main__":
    # 创建一个简单的参数对象来测试
    class Args:
        def __init__(self):
            # 基础参数
            self.data = 'DEAP'  # 或 'SEED'
            self.root_path = r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python"
            self.seq_len = 256
            self.batch_size = 32
            self.num_workers = 0
            self.embed = 'timeF'

            # EEG特定参数
            self.num_class = 2
            self.classification_type = 'valence'
            self.overlap = 128
            self.normalize = True
            self.filter_freq = (0.5, 45)
            self.sampling_rate = 128
            self.subject_list = ['s01', 's02']  # 测试用少量被试

            # 其他可能需要的参数
            self.pred_len = 0
            self.label_len = 0


    # 测试DEAP数据集
    print("测试DEAP数据集的data_provider...")
    args = Args()
    args.data = 'DEAP'

    train_data, train_loader = data_provider(args, 'train')
    print(f"训练集大小: {len(train_data)}")
    print(f"批次数量: {len(train_loader)}")

    # 获取一个批次
    for batch in train_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        print(f"\n批次形状:")
        print(f"  - batch_x: {batch_x.shape}")
        print(f"  - batch_y: {batch_y.shape}")
        print(f"  - batch_x_mark: {batch_x_mark.shape}")
        print(f"  - batch_y_mark: {batch_y_mark.shape}")
        break

    # 测试SEED数据集
    print("\n" + "=" * 50)
    print("测试SEED数据集的data_provider...")
    args.data = 'SEED'
    args.root_path = r"D:\文件\文件\HKU\Dissertation\dataset\SEED\SEED\Preprocessed_EEG"
    args.num_class = 3
    args.filter_freq = (0.5, 50)
    args.sampling_rate = 200
    args.subject_list = ['1_20131027.mat']

    train_data, train_loader = data_provider(args, 'train')
    print(f"训练集大小: {len(train_data)}")
    print(f"批次数量: {len(train_loader)}")
