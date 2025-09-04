
import subprocess
import time
import os

# 测试配置 - 针对DEAP优化
models_to_test = [
    {'name': 'GPT2', 'llm_model': 'GPT2', 'llm_dim': 768, 'batch_size': 32},
    {'name': 'GPT2-MEDIUM', 'llm_model': 'GPT2-MEDIUM', 'llm_dim': 1024, 'batch_size': 24},
    {'name': 'GPT2-LARGE', 'llm_model': 'GPT2-LARGE', 'llm_dim': 1280, 'batch_size': 16},
    {'name': 'BERT', 'llm_model': 'BERT', 'llm_dim': 768, 'batch_size': 32},
    {'name': 'ROBERTA', 'llm_model': 'ROBERTA', 'llm_dim': 768, 'batch_size': 32},
]

# 基础参数 - DEAP配置
base_cmd = [
    'python', 'run_main_enhanced.py',
    '--is_training', '1',
    '--model', 'TimeLLM',
    '--task_name', 'classification',
    '--data', 'DEAP',
    '--root_path', '/content/drive/MyDrive/EEG_Data/DEAP',
    '--num_class', '2',  # DEAP是二分类
    '--enc_in', '32',    # DEAP只有32个通道
    '--dec_in', '32', 
    '--c_out', '32',
    '--seq_len', '256',
    '--loss_type', 'focal',  # DEAP推荐用focal loss
    
    
    '--learning_rate', '0.0001',
    '--train_epochs', '2',  # 快速测试只跑2个epoch
    '--warmup_epochs', '0',  # 测试时不需要warmup
    '--d_model', '32',
    '--d_ff', '64',
    '--patch_len', '16',
    '--stride', '8',
    '--use_amp',  # 启用混合精度
    '--num_workers', '2',  # 加速数据加载
    '--patience', '5'  # 减少早停耐心值
]

# 创建结果记录文件
results_file = '/content/drive/MyDrive/Time-LLM-EEG/model_comparison_results.txt'
os.makedirs(os.path.dirname(results_file), exist_ok=True)

print("开始在DEAP数据集上测试不同的LLM模型...")
print("预计每个模型测试时间：10-20分钟")

with open(results_file, 'w') as f:
    f.write("模型比较测试结果（DEAP数据集）\n")
    f.write(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")

for i, model_config in enumerate(models_to_test):
    print(f"\n{'='*60}")
    print(f"测试模型 [{i+1}/{len(models_to_test)}]: {model_config['name']}")
    print(f"Batch Size: {model_config['batch_size']}")
    print(f"{'='*60}")
    
    model_id = f"test_DEAP_{model_config['name']}_{int(time.time())}"
    
    # 构建命令
    cmd = base_cmd + [
        '--model_id', model_id,
        '--llm_model', model_config['llm_model'],
        '--llm_dim', str(model_config['llm_dim']),
        '--batch_size', str(model_config['batch_size']),
        '--eval_batch_size', str(min(model_config['batch_size'], 16))
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    try:
        # 将输出保存到文件
        log_file = f'/content/timellm/Time-LLM-main/logs/{model_id}.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(cmd, 
                                     cwd='/content/timellm/Time-LLM-main',
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     text=True)
            
            # 实时显示输出
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                
                # 提取关键指标
                if "Epoch: 2" in line and "Vali Acc:" in line:
                    # 保存最终结果
                    with open(results_file, 'a') as f:
                        f.write(f"\n{model_config['name']}:\n")
                        f.write(f"  - 训练时间: {time.time()-start_time:.1f}秒\n")
                        f.write(f"  - 最终结果: {line}")
                        f.write(f"  - Batch Size: {model_config['batch_size']}\n")
            
            process.wait()
            
    except KeyboardInterrupt:
        print(f"\n跳过 {model_config['name']}")
        with open(results_file, 'a') as f:
            f.write(f"\n{model_config['name']}: 跳过\n")
        continue
    except Exception as e:
        print(f"\n错误: {e}")
        with open(results_file, 'a') as f:
            f.write(f"\n{model_config['name']}: 错误 - {e}\n")
        continue
    
    # 显示当前结果
    print(f"\n{model_config['name']} 测试完成！用时：{time.time()-start_time:.1f}秒")
    
    # 等待GPU冷却
    if i < len(models_to_test) - 1:
        print("\n等待5秒让GPU冷却...")
        time.sleep(5)

print(f"\n{'='*60}")
print("所有模型测试完成！")
print(f"结果保存在: {results_file}")
print(f"{'='*60}")

# 显示结果摘要
print("\n结果摘要：")
with open(results_file, 'r') as f:
    print(f.read())
