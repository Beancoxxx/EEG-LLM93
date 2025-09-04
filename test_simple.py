
import subprocess

print("测试GPT2模型（已验证可工作）...")

cmd = [
    'python', 'run_main_enhanced.py',
    '--is_training', '1',
    '--model', 'TimeLLM',
    '--model_id', 'test_GPT2_simple',
    '--task_name', 'classification',
    '--data', 'DEAP',
    '--root_path', '/content/drive/MyDrive/EEG_Data/DEAP',
    '--num_class', '2',
    '--enc_in', '32',
    '--dec_in', '32', 
    '--c_out', '32',
    '--seq_len', '256',
    '--loss_type', 'weighted_ce',  # 改用weighted_ce避免focal loss问题
    '--learning_rate', '0.0001',
    '--train_epochs', '2',
    '--warmup_epochs', '0',
    '--llm_model', 'GPT2',
    '--llm_dim', '768',
    '--llm_layers', '2',
    '--d_model', '32',
    '--d_ff', '64',
    '--patch_len', '16',
    '--stride', '8',
    '--batch_size', '32',
    '--use_amp',
    '--num_workers', '2'
]

subprocess.run(cmd, cwd='/content/timellm/Time-LLM-main')
