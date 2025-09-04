"""
修改版run_main.py - 支持EEG情绪分类任务
主要修改：
1. 添加num_class参数
2. 支持分类任务的损失函数
3. 修改train和vali函数支持分类
4. 添加分类评估指标
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

from utils.tools import EarlyStopping, adjust_learning_rate, load_content
#from utils.dtw_metric import dtw, accelerated_dtw
from utils.tools import del_files


parser = argparse.ArgumentParser(description='Time-LLM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, classification]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TimeLLM',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 分类任务特定参数
parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification task')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length') 
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
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
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# LLM config
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768

# Model config
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--temperature', type=int, default=10, help='temperature to divide in Patch Reprograming')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')

args = parser.parse_args()

# 设置加速器
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# 对于分类任务，可能不需要DeepSpeed
if args.task_name == 'classification':
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
else:
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


def vali_classification(args, accelerator, model, vali_data, vali_loader, criterion):
    """分类任务的验证函数"""
    total_loss = []
    preds = []
    trues = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.long().to(accelerator.device)  # 分类标签是long类型
            
            # 模型预测
            outputs = model(batch_x, batch_x_mark)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
            
            # 收集预测结果用于计算准确率
            preds.append(outputs.detach())
            trues.append(batch_y.detach())
    
    # 聚合所有批次的结果
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    
    # 计算准确率
    _, predicted = torch.max(preds, 1)
    accuracy = (predicted == trues).float().mean()
    
    # 计算平均损失
    total_loss = np.average(total_loss)
    
    model.train()
    return total_loss, accuracy.item()


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
        setting = '{}_{}_{}_{}_ft{}_sl{}_nc{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.num_class,  # 使用num_class替代pred_len
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

    time_now = time.time()
    train_steps = len(train_loader)
    
    # 早停策略 - 分类任务使用不同的监控指标
    if args.task_name == 'classification':
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, 
                                     verbose=True, mode='max')  # 监控准确率（越高越好）
    else:
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # 优化器
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # 学习率调度器
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    # 损失函数
    if args.task_name == 'classification':
        criterion = nn.CrossEntropyLoss()
        accelerator.print("使用CrossEntropyLoss进行分类训练")
    else:
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

    # 准备加速器
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        
        model.train()
        epoch_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            
            if args.task_name == 'classification':
                # 分类任务
                batch_y = batch_y.long().to(accelerator.device)  # 标签是long类型
                outputs = model(batch_x, batch_x_mark)
                loss = criterion(outputs, batch_y)
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
                accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        
        # 验证和测试
        if args.task_name == 'classification':
            vali_loss, vali_acc = vali_classification(args, accelerator, model, vali_data, vali_loader, criterion)
            test_loss, test_acc = vali_classification(args, accelerator, model, test_data, test_loader, criterion)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Vali Acc: {3:.4f} Test Loss: {4:.7f} Test Acc: {5:.4f}".format(
                    epoch + 1, train_loss, vali_loss, vali_acc, test_loss, test_acc))
            
            # 早停使用准确率
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

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    del_files(path)
    accelerator.print('success delete checkpoints')