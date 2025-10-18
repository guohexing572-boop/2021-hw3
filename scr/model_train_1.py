import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
import torch.optim as optim
from tools.hw3_model import resnet18
from tools.hw3_common_tools import plot_loss_curves, plot_accuracy_curves, plot_training_curves
from PIL import Image
import torchvision.transforms as transforms

# 设置基础路径和设备
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

if __name__ == "__main__":
    # ============================ 配置参数 ============================
    # 数据集路径
    parent_dir = os.path.dirname(BASE_DIR)  # 获取上级文件夹


    # 创建日志目录，以当前时间命名
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练超参数
    MAX_EPOCH = 256  # 总训练轮数
    BATCH_SIZE = 128  # 批大小
    LR = 0.001  # 初始学习率
    log_interval = 1  # 日志记录间隔
    val_interval = 1  # 验证间隔
    VAL_RATIO = 0.2  # 验证集比例


    # ============================ step 1/5 数据加载 ============================
    # It is important to do data augmentation in training.
    # However, not every augmentation is useful.
    # Please think about what kind of augmentation is helpful for food recognition.
    train_tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
    ])

    # We don't need augmentations in testing and validation.
    # All we need here is to resize the PIL image and transform it into Tensor.
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=train_tfm)
    test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)


    # 应用不同的transform
    # train_data = TransformedSubset(train_set, transform=train_transform)
    # valid_data = TransformedSubset(val_dataset, transform=valid_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


    print(f"训练集: {len(train_set)}")
    print(f"验证集: {len(valid_set)}")


    # ============================ step 2/5 模型定义 ============================

    model = resnet18(num_classes = 11)
    model.to(device)  # 将模型移动到设备（GPU/CPU）

    # 打印模型信息
    print(f"模型已创建，移动到设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失

    # ============================ step 4/5 优化器 ============================
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # ============================ step 5/5 训练循环 ============================
    # 记录训练过程中的各项指标
    train_losses = []
    val_losses = []
    train_accuracies = []  # 新增：训练准确率记录
    val_accuracies = []  # 新增：验证准确率记录
    learning_rates = []  # 新增：学习率记录
    best_val_accuracy = 0.0  # 记录最佳验证准确率

    print("开始训练...")
    for epoch in range(MAX_EPOCH):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 确保标签是long类型（CrossEntropyLoss要求）
            if target.dtype != torch.long:
                target = target.long()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)  # 记录训练准确率

        # 更新学习率并记录
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        scheduler.step()

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)

                # 确保标签是long类型
                if target.dtype != torch.long:
                    target = target.long()

                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = 100.0 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)  # 记录验证准确率

        # 打印训练和验证信息
        if epoch % log_interval == 0:
            print(f'Epoch: {epoch:03d}/{MAX_EPOCH}, '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
                  f'LR: {current_lr:.6f}')

        # 保存最佳模型（基于验证准确率）
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "train_loss": avg_train_loss
            }
            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
            print(f"✅ 保存最佳模型，验证准确率: {best_val_accuracy:.2f}%")

    # ===== 训练结束 =====
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(f"训练完成时间: {time_str}")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")

    # 保存训练记录
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy
    }
    torch.save(training_history, os.path.join(log_dir, 'training_history.pth'))

    # 绘制各种曲线
    picture_path_loss = os.path.join(log_dir, 'loss_curves.png')
    picture_path_acc = os.path.join(log_dir, 'accuracy_curves.png')
    picture_path_combined = os.path.join(log_dir, 'training_curves.png')

    plot_loss_curves(train_losses, val_losses, picture_path_loss)
    plot_accuracy_curves(train_accuracies, val_accuracies, picture_path_acc)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, picture_path_combined)

    print(f"损失曲线已保存至: {picture_path_loss}")
    print(f"准确率曲线已保存至: {picture_path_acc}")
    print(f"综合训练曲线已保存至: {picture_path_combined}")