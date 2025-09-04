"""
增强版run_main.py - 集成了所有训练辅助模块
主要改进：
1. 集成评估指标模块（F1分数、混淆矩阵等）
2. 集成可视化工具（训练曲线、特征可视化）
3. 集成高级损失函数（Focal Loss、加权CE等）
4. 集成训练辅助工具（更好的学习率调度、模型保存等）
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from models import Autoformer, DLinear, TimeLLM

# 原始工具
from utils.tools import EarlyStopping, adjust_learning_rate, load_content
from utils.tools import del_files

# 新增的训练辅助模块
from utils.metrics_classification import EmotionMetrics, compute_batch_metrics
from utils.tools_classification import TrainingVisualizer, FeatureVisualizer, create_experiment_summary
from utils.loss_classification import create_loss_function, compute_class_weights
from utils.train_utils_classification import (
    create_optimizer, create_scheduler,
    ModelCheckpoint, GradientClipping, TrainingLogger,
    WarmupCosineAnnealingLR
)

parser = argparse.ArgumentParser(description='Time-LLM Enhanced')

# basic config
parser.add_argument('--task_name', type=str, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, classification]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test_improved_gpt2', help='model id')
parser.add_argument('--model', type=str, default='TimeLLM',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')

# data loader
parser.add_argument('--data', type=str, default='DEAP', help='dataset type')
parser.add_argument('--root_path', type=str, default='./DEAP/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 分类任务特定参数
parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification task')
parser.add_argument('--classification_type', type=str, default='valence', 
                    help='classification type for EEG: valence, arousal, or both')
parser.add_argument('--loss_type', type=str, default='focal', 
                    help='loss function type: ce, weighted_ce, focal, label_smoothing')
parser.add_argument('--focal_alpha', type=float, default=0.75, help='focal loss alpha')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing factor')

# EEG特定参数
parser.add_argument('--subject_list', type=str, nargs='*', default=None,
                    help='list of subjects to use, e.g., --subject_list s01 s02 s03')
parser.add_argument('--overlap', type=int, default=128,
                    help='overlap for sliding window (0 to seq_len-1)')
parser.add_argument('--normalize', type=bool, default=True,
                    help='whether to normalize EEG data')
parser.add_argument('--filter_freq', type=float, nargs=2, default=None,
                    help='bandpass filter frequency range, e.g., --filter_freq 0.5 45')
parser.add_argument('--sampling_rate', type=int, default=128,
                    help='sampling rate of EEG data (DEAP: 128, SEED: 200)')

# EEG通道选择参数 - 新增！
parser.add_argument('--use_channel_selection', type=bool, default=True,
                    help='whether to use emotion-related channel selection')
parser.add_argument('--channel_selection', type=str, default='comprehensive_emotion',
                    help='''channel selection strategy. Options:
                    - auto: automatically select based on classification_type
                    - frontal_emotion: frontal regions for emotion regulation
                    - frontal_asymmetry: frontal asymmetry for valence
                    - temporal_emotion: temporal regions for emotion memory
                    - parietal_attention: parietal regions for attention
                    - central_motor: central regions for motor preparation
                    - comprehensive_emotion: comprehensive emotion network
                    - valence_specific: optimized for valence classification
                    - arousal_specific: optimized for arousal classification''')
parser.add_argument('--print_channel_info', action='store_true',
                    help='print available channel selection strategies')

# forecasting task
parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length') 
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=32, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=32, help='decoder input size')
parser.add_argument('--c_out', type=int, default=32, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='focal', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=2)
parser.add_argument('--percent', type=int, default=100)

# 新增的训练辅助参数
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer type: adam, adamw, sgd')
parser.add_argument('--scheduler_type', type=str, default='cosine', 
                    help='scheduler type: cosine, step, exponential, reduce_on_plateau')
parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# LLM config
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default=768, help='LLM model dimension')# LLama7b:4096; GPT2-small:768

# Model config
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--temperature', type=int, default=10, help='temperature to divide in Patch Reprograming')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')

args = parser.parse_args()

if args.print_channel_info and args.data == 'DEAP':
    from data_provider.data_factory import print_available_channel_groups, get_recommended_channel_selection
    print_available_channel_groups()
    if args.task_name == 'classification':
        recommended = get_recommended_channel_selection(args.classification_type, args.num_class)
        print(f"\n基于当前配置的推荐策略: {recommended}")
    exit()

# 设置加速器
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# 对于分类任务，可能不需要DeepSpeed
if args.task_name == 'classification':
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
else:
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    
if args.data in ['DEAP', 'SEED'] and args.task_name == 'classification':
    accelerator.print("="*80)
    accelerator.print("EEG情绪分类任务配置:")
    accelerator.print("="*80)
    accelerator.print(f"数据集: {args.data}")
    accelerator.print(f"分类类型: {args.classification_type}")
    accelerator.print(f"分类数量: {args.num_class}")
    accelerator.print(f"序列长度: {args.seq_len}")
    accelerator.print(f"滑动窗口重叠: {args.overlap}")
    accelerator.print(f"滤波频段: {args.filter_freq}")
    accelerator.print(f"采样率: {args.sampling_rate}")
    accelerator.print(f"使用通道选择: {args.use_channel_selection}")
    accelerator.print(f"通道选择策略: {args.channel_selection}")
    if args.subject_list:
        accelerator.print(f"指定被试: {args.subject_list}")
    accelerator.print("="*80)


def vali_classification_enhanced(args, accelerator, model, vali_data, vali_loader, criterion, metrics_tracker):
    """增强版分类验证函数 - 包含详细指标"""
    total_loss = []
    metrics_tracker.reset()
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.long().to(accelerator.device)
            
            # 模型预测
            outputs = model(batch_x, batch_x_mark)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
            
            # 预测概率和类别
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # 更新指标跟踪器
            metrics_tracker.update(preds, batch_y, probs)
    
    # 计算所有指标
    total_loss = np.average(total_loss)
    all_metrics = metrics_tracker.compute_metrics()
    
    model.train()
    return total_loss, all_metrics


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """原始的验证函数（用于预测任务）"""
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            loss = criterion(outputs, batch_y)
            mae_loss = mae_metric(outputs, batch_y)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss


# 主训练循环
for ii in range(args.itr):
    # 设置实验记录
    if args.task_name == 'classification':
        # 为EEG数据集添加通道选择信息到实验名称
        if args.data in ['DEAP', 'SEED'] and args.use_channel_selection:
            setting = '{}_{}_{}_{}_ft{}_sl{}_nc{}_cs{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.num_class,
                args.channel_selection,  # 添加通道选择策略到实验名称
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.batch_size,
                args.des, ii)
        else:
            setting = '{}_{}_{}_{}_ft{}_sl{}_nc{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.num_class,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.batch_size,
                args.des, ii)
    else:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

    # 数据加载
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 对于EEG数据集，打印实际使用的通道信息
    if args.data in ['DEAP', 'SEED'] and hasattr(train_data, 'get_channel_info'):
        channel_info = train_data.get_channel_info()
        accelerator.print(f"\n实际使用的通道信息:")
        accelerator.print(f"  - 通道数量: {channel_info['n_channels']}")
        accelerator.print(f"  - 通道名称: {', '.join(channel_info['channel_names'])}")
        accelerator.print(f"  - 模型输入维度已更新为: {args.enc_in}")

    # 模型初始化
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    # 检查点路径
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    # 创建训练辅助工具
    if args.task_name == 'classification':
        # 评估指标跟踪器
        train_metrics = EmotionMetrics(args.data)
        val_metrics = EmotionMetrics(args.data)
        test_metrics = EmotionMetrics(args.data)
        
        # 可视化工具
        visualizer = TrainingVisualizer(os.path.join(path, 'visualizations'))
        
        # 计算类别权重
        accelerator.print("计算类别权重...")
        train_labels = []
        for batch_x, batch_y, _, _ in train_loader:
            train_labels.extend(batch_y.cpu().numpy())
        class_weights = compute_class_weights(train_labels)
        accelerator.print(f"类别权重: {class_weights}")
        
        # 创建损失函数
        if args.loss_type == 'focal':
            criterion = create_loss_function(
                'focal',
                num_classes=args.num_class,
                alpha=args.focal_alpha,
                gamma=args.focal_gamma
            )
        elif args.loss_type == 'label_smoothing':
            criterion = create_loss_function(
                'label_smoothing',
                num_classes=args.num_class,
                smoothing=args.label_smoothing
            )
        else:
            criterion = create_loss_function(
                args.loss_type,
                num_classes=args.num_class,
                class_weights=class_weights.tolist()
            )
        accelerator.print(f"使用损失函数: {args.loss_type}")
    else:
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

    # 优化器和调度器（使用新的工具）
    model_optim = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练步数
    train_steps = len(train_loader)
    
    # 学习率调度器
    if args.lradj == 'COS':
        scheduler = WarmupCosineAnnealingLR(
            model_optim,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.train_epochs
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )

    # 早停和模型保存
    if args.task_name == 'classification':
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, 
                                     verbose=True, mode='max')
        checkpoint = ModelCheckpoint(
            path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_last=True
        )
    else:
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
        checkpoint = ModelCheckpoint(
            path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_last=True
        )
    
    # 梯度裁剪和日志记录
    gradient_clipper = GradientClipping(max_norm=args.grad_clip)
    logger = TrainingLogger(os.path.join(path, 'logs'), setting)

    # 准备加速器
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    for epoch in range(args.train_epochs):
        iter_count = 0
        time_now = time.time()
        train_loss = []
        train_batch_metrics = []
        
        model.train()
        epoch_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            
            if args.task_name == 'classification':
                # 分类任务
                batch_y = batch_y.long().to(accelerator.device)
                outputs = model(batch_x, batch_x_mark)
                loss = criterion(outputs, batch_y)
                
                # 计算批次指标
                if (i + 1) % 100 == 0:
                  batch_metrics = compute_batch_metrics(outputs, batch_y)
                  train_batch_metrics.append(batch_metrics)
            else:
                # 预测任务
                batch_y = batch_y.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
                
                # encoder - decoder
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                
                loss = criterion(outputs, batch_y)

            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                if args.task_name == 'classification' and train_batch_metrics:
                    avg_acc = np.mean([m['accuracy'] for m in train_batch_metrics[-100:]])
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | acc: {3:.4f}".format(
                            i + 1, epoch + 1, loss.item(), avg_acc))
                else:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()))
                
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                grad_norm = gradient_clipper(model)
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                grad_norm = gradient_clipper(model)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        
        # 训练指标汇总
        train_epoch_metrics = {'loss': train_loss}
        if args.task_name == 'classification' and train_batch_metrics:
            train_epoch_metrics['accuracy'] = np.mean([m['accuracy'] for m in train_batch_metrics])
            train_epoch_metrics['f1_score'] = np.mean([m['f1_score'] for m in train_batch_metrics])
        
        # 验证和测试
        if args.task_name == 'classification':
            vali_loss, vali_metrics_dict = vali_classification_enhanced(
                args, accelerator, model, vali_data, vali_loader, criterion, val_metrics)
            test_loss, test_metrics_dict = vali_classification_enhanced(
                args, accelerator, model, test_data, test_loader, criterion, test_metrics)
            
            # 提取关键指标
            vali_acc = vali_metrics_dict['accuracy']
            test_acc = test_metrics_dict['accuracy']
            vali_f1 = vali_metrics_dict['f1_macro']
            test_f1 = test_metrics_dict['f1_macro']
            
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Acc: {2:.4f} | "
                "Vali Loss: {3:.7f} Acc: {4:.4f} F1: {5:.4f} | "
                "Test Loss: {6:.7f} Acc: {7:.4f} F1: {8:.4f}".format(
                    epoch + 1, train_loss, train_epoch_metrics.get('accuracy', 0),
                    vali_loss, vali_acc, vali_f1,
                    test_loss, test_acc, test_f1))
            
            # 更新可视化
            val_epoch_metrics = {'loss': vali_loss, 'accuracy': vali_acc, 'f1_score': vali_f1}
            visualizer.update(
                epoch=epoch + 1,
                train_metrics=train_epoch_metrics,
                val_metrics=val_epoch_metrics,
                lr=model_optim.param_groups[0]['lr']
            )
            
            # 记录日志
            logger.log_metrics(epoch + 1, train_epoch_metrics, prefix='train_')
            logger.log_metrics(epoch + 1, val_epoch_metrics, prefix='val_')
            
            # 模型保存
            checkpoint(epoch + 1, model, model_optim, val_epoch_metrics)
            
            # 早停
            early_stopping(vali_acc, model, path)
        else:
            vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))
            
            # 早停使用损失
            early_stopping(vali_loss, model, path)
            
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # 训练结束后的处理
    if args.task_name == 'classification':
        # 保存可视化结果
        visualizer.plot_training_curves()
        visualizer.save_history()
        
        # 在测试集上的最终评估
        accelerator.print("\n" + "="*60)
        accelerator.print("最终测试集评估")
        accelerator.print("="*60)
        
        # 加载最佳模型
        best_model_path = os.path.join(path, 'best_checkpoint.pth')
        if os.path.exists(best_model_path):
            checkpoint_data = torch.load(best_model_path, map_location=accelerator.device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            accelerator.print("已加载最佳模型")
        
        # 测试集评估
        test_loss, test_metrics_dict = vali_classification_enhanced(
            args, accelerator, model, test_data, test_loader, criterion, test_metrics)
        
        # 打印详细结果
        test_metrics.print_emotion_summary()
        
        # 保存混淆矩阵
        test_metrics.plot_confusion_matrix(
            save_path=os.path.join(path, 'confusion_matrix.png')
        )
        
        # 创建实验报告
        config = vars(args)
        results = {
            'best_val_accuracy': checkpoint.best_value if checkpoint.best_value else 0,
            'test_accuracy': test_metrics_dict['accuracy'],
            'test_f1_macro': test_metrics_dict['f1_macro'],
            'test_f1_weighted': test_metrics_dict['f1_weighted'],
            'total_epochs': epoch + 1
        }
        
        create_experiment_summary(
            experiment_name=setting,
            config=config,
            results=results,
            save_dir=path
        )
        
        # 保存日志历史
        logger.save_history()

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    del_files(path)
    accelerator.print('success delete checkpoints')
