```python
# train_eeg_classification.py
import os
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from google.colab import drive
from data_provider.data_factory import data_provider
from models.TimeLLM import Model

def mount_drive(path='/content/drive'):
    drive.mount(path)
    return path

class TrainConfig:
    # 选择数据集: 'DEAP' 或 'SEED'
    dataset = 'DEAP'  # 或 'SEED'

    # 数据路径
    data_path = f"/content/drive/MyDrive/EEG_Data/{dataset}/data_preprocessed_python"
    seq_len = 256
    pred_len = 0
    label_len = 0

    # 通用训练参数
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-3
    num_epochs = 20

    # 模型保存
    checkpoint_dir = "/content/drive/MyDrive/checkpoints/"

    # 数据集特定参数
    num_class = 2 if dataset == 'DEAP' else 3
    classification_type = 'valence' if dataset == 'DEAP' else None
    overlap = 128
    normalize = True
    filter_freq = (0.5, 45)
    sampling_rate = 128 if dataset == 'DEAP' else 200
    subject_list = None  # 默认分配

    # LLM配置可选
    llm_model = 'GPT2'  # 或 'LLAMA'
    llm_layers = 2
    d_model = 32
    n_heads = 4
    d_ff = 64
    dropout = 0.1

    # 其它TimeLLM配置
    features = 'M'
    embed = 'timeF'
    freq = 'h'


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for x, y, x_mark, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, x_mark.to(device))
        loss = criterion(logits, y.long())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return sum(losses)/len(losses), acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, x_mark, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, x_mark.to(device))
            loss = criterion(logits, y.long())
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return sum(losses)/len(losses), acc, f1, cm


def save_checkpoint(state, is_best, checkpoint_dir, filename='last.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)


def main():
    # 1. 挂载Drive
    mount_drive()

    # 2. 配置 & GPU 检测
    cfg = TrainConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用GPU: {gpu_name}")
        if 'A100' in gpu_name:
            cfg.batch_size = 32
            print(f"检测到A100，自动将batch_size调整为{cfg.batch_size}")

    # 3. 数据加载
    data_args = cfg
    train_data, train_loader = data_provider(data_args, 'train')
    val_data, val_loader = data_provider(data_args, 'val')
    test_data, test_loader = data_provider(data_args, 'test')

    # 4. 模型 & 优化器
    model_cfg = cfg
    model = Model(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

    # 5. 训练循环
    best_val_f1 = 0.0
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, device)
        is_best = val_f1 > best_val_f1
        best_val_f1 = max(val_f1, best_val_f1)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f} |"
              f" Val loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_f1': best_val_f1,
        }, is_best, cfg.checkpoint_dir)

    # 6. 测试集评估
    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion, device)
    print(f"Test Performance: loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

if __name__ == '__main__':
    main()
```
