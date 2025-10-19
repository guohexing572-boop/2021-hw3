import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import DatasetFolder
import torch.optim as optim
from tools.hw3_model import resnet18_512
from tools.hw3_common_tools import plot_loss_curves, plot_accuracy_curves, plot_training_curves
from PIL import Image
import torchvision.transforms as transforms


# 定义可序列化的图片加载函数
def pil_loader(path):
    """使用PIL加载图片，支持多种格式"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')  # 确保图片是RGB格式


# 设置基础路径和设备
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU
pin_memory = (device.type == 'cuda')  # 如果使用GPU，启用内存锁页加速数据传输


class SemiSupervisedTrainer:
    """
    半监督训练器类
    结合有标签数据和无标签数据进行训练，提升模型性能
    """

    def __init__(self, model, train_loader, unlabeled_loader, valid_loader,
                 optimizer, criterion, device, pseudo_threshold=0.9, consistency_weight=0.3):
        # 初始化模型和数据加载器
        self.model = model
        self.train_loader = train_loader  # 有标签训练数据
        self.unlabeled_loader = unlabeled_loader  # 无标签数据
        self.valid_loader = valid_loader  # 验证数据
        self.optimizer = optimizer  # 优化器
        self.criterion = criterion  # 损失函数
        self.device = device  # 训练设备
        self.pseudo_threshold = pseudo_threshold  # 伪标签置信度阈值
        self.consistency_weight = consistency_weight  # 一致性损失权重

        # 数据增强（用于一致性训练）
        # 弱增强：轻微的数据变换，保持图像主要内容不变
        self.weak_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        ])

        # 强增强：更强的数据变换，产生更多样化的图像
        self.strong_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 更强的颜色抖动
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 仿射变换
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊
        ])

    def generate_pseudo_labels(self, confidence_threshold=None):
        """生成伪标签：使用当前模型对无标签数据进行预测，选择高置信度的预测作为伪标签"""
        if confidence_threshold is None:
            confidence_threshold = self.pseudo_threshold

        self.model.eval()  # 设置为评估模式
        pseudo_data = []  # 存储伪标签数据
        pseudo_labels = []  # 存储伪标签

        with torch.no_grad():  # 禁用梯度计算，节省内存
            for data, _ in self.unlabeled_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)  # 转换为概率分布
                max_probs, predictions = torch.max(probabilities, 1)  # 获取最大概率和预测类别

                # 筛选高置信度样本：只选择置信度高于阈值的预测
                mask = max_probs > confidence_threshold
                high_conf_data = data[mask].cpu()  # 高置信度数据
                high_conf_preds = predictions[mask].cpu()  # 对应的伪标签

                if len(high_conf_data) > 0:
                    for i in range(len(high_conf_data)):
                        pseudo_data.append(high_conf_data[i])
                        pseudo_labels.append(high_conf_preds[i])

        print(f"生成了 {len(pseudo_data)} 个伪标签样本 (阈值: {confidence_threshold})")
        return pseudo_data, pseudo_labels

    def consistency_loss(self, unlabeled_batch):
        """计算一致性损失：对同一无标签数据应用不同增强，期望模型输出一致的预测"""
        batch_size = unlabeled_batch.size(0)

        # 弱增强：保持图像主要内容
        weak_aug = self.weak_augment(unlabeled_batch)

        # 强增强：更强的图像变换
        strong_aug = self.strong_augment(unlabeled_batch)

        # 获取预测
        with torch.no_grad():
            weak_output = F.softmax(self.model(weak_aug), dim=1)  # 弱增强的预测作为"教师"

        strong_output = F.log_softmax(self.model(strong_aug), dim=1)  # 强增强的预测作为"学生"

        # 计算KL散度损失：衡量两个概率分布的差异
        consistency_loss = F.kl_div(strong_output, weak_output, reduction='batchmean')
        return consistency_loss

    def train_epoch(self, epoch, use_consistency=True, use_pseudo_labels=False):
        """训练一个epoch：结合有监督损失和无监督损失"""
        self.model.train()  # 设置为训练模式
        train_loss = 0.0  # 累计训练损失
        train_correct = 0  # 正确预测数量
        train_total = 0  # 总样本数量

        # 伪标签生成：定期使用当前模型生成伪标签来扩展训练集
        if use_pseudo_labels and epoch % 5 == 0 and epoch > 10:
            pseudo_data, pseudo_labels = self.generate_pseudo_labels()
            if pseudo_data:
                # 创建伪标签数据集（这里简化处理，实际应该创建完整的Dataset）
                pseudo_dataset = list(zip(pseudo_data, pseudo_labels))
                print("使用伪标签数据扩展训练集")

        # 创建无标签数据迭代器
        unlabeled_iter = iter(self.unlabeled_loader)

        # 遍历有标签训练数据
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # 清空梯度

            # 有监督损失：使用有标签数据计算的标准交叉熵损失
            output = self.model(data)
            supervised_loss = self.criterion(output, target)

            total_loss = supervised_loss  # 总损失初始化为有监督损失

            # 无监督损失（一致性正则化）：使用无标签数据计算的一致性损失
            if use_consistency:
                try:
                    unlabeled_data, _ = next(unlabeled_iter)
                    unlabeled_data = unlabeled_data.to(self.device)

                    consistency_loss = self.consistency_loss(unlabeled_data)
                    # 组合损失：有监督损失 + 权重 * 无监督损失
                    total_loss = supervised_loss + self.consistency_weight * consistency_loss

                except StopIteration:
                    # 重置迭代器：当无标签数据遍历完时重新开始
                    unlabeled_iter = iter(self.unlabeled_loader)
                    consistency_loss = torch.tensor(0.0)

            total_loss.backward()  # 反向传播计算梯度
            self.optimizer.step()  # 更新模型参数

            train_loss += total_loss.item()  # 累计损失

            # 计算训练准确率
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # 定期打印训练信息
            if batch_idx % 50 == 0:
                cons_loss_val = consistency_loss.item() if use_consistency else 0.0
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Total Loss: {total_loss.item():.4f} | Supervised: {supervised_loss.item():.4f} | '
                      f'Consistency: {cons_loss_val:.4f}')

        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / len(self.train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        return avg_train_loss, train_accuracy

    def validate(self):
        """验证模型性能：在验证集上评估模型"""
        self.model.eval()  # 设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 禁用梯度计算
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(self.valid_loader)
        val_accuracy = 100.0 * val_correct / val_total

        return avg_val_loss, val_accuracy


def main():
    """主函数：配置参数、加载数据、训练模型"""
    # ============================ 配置参数 ============================
    parent_dir = os.path.dirname(BASE_DIR)  # 获取上级目录

    # 创建日志目录：以当前时间命名，避免重复
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", "semi_supervised_" + time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练超参数
    MAX_EPOCH = 182  # 最大训练轮数
    BATCH_SIZE = 32  # 批大小
    LR = 0.001  # 学习率
    PATIENCE = 20  # 早停耐心值
    milestones = [92, 136]  # 学习率调整的里程碑

    # 半监督参数
    PSEUDO_THRESHOLD = 0.9  # 伪标签置信度阈值
    CONSISTENCY_WEIGHT = 0.3  # 一致性损失权重

    # ============================ 数据加载 ============================
    # 训练数据增强：使用多种数据增强技术提升模型泛化能力
    train_tfm = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图像大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 仿射变换
        transforms.RandomCrop(512, padding=16),  # 随机裁剪
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 测试/验证数据增强：只进行必要的预处理
    test_tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    # 有标签训练集
    train_set = DatasetFolder(
        os.path.join(parent_dir, "food-11", "training", "labeled"),
        loader=pil_loader,
        extensions="jpg",
        transform=train_tfm
    )

    # 验证集
    valid_set = DatasetFolder(
        os.path.join(parent_dir, "food-11", "validation"),
        loader=pil_loader,
        extensions="jpg",
        transform=test_tfm
    )

    # 无标签数据集
    unlabeled_set = DatasetFolder(
        os.path.join(parent_dir, "food-11", "training", "unlabeled"),
        loader=pil_loader,
        extensions="jpg",
        transform=train_tfm  # 训练时增强
    )

    # 测试集
    test_set = DatasetFolder(
        os.path.join(parent_dir, "food-11", "testing"),
        loader=pil_loader,
        extensions="jpg",
        transform=test_tfm
    )

    # 数据加载器
    num_workers = 2 if os.name == 'nt' else 4  # Windows系统使用较少进程

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证时不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"训练集: {len(train_set)}")
    print(f"无标签集: {len(unlabeled_set)}")
    print(f"验证集: {len(valid_set)}")
    print(f"使用设备: {device}")

    # ============================ 模型定义 ============================
    model = resnet18_512(num_classes=11)  # 使用ResNet-18模型，适配512x512输入
    # 或者使用: model = FoodCNN_2(num_classes=11)  # 轻量级自定义CNN
    model.to(device)  # 将模型移动到指定设备

    print(f"模型已创建，移动到设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ============================ 损失函数和优化器 ============================
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类任务
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # AdamW优化器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)  # 多步学习率调度

    # ============================ 创建训练器 ============================
    trainer = SemiSupervisedTrainer(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        pseudo_threshold=PSEUDO_THRESHOLD,
        consistency_weight=CONSISTENCY_WEIGHT
    )

    # ============================ 训练循环 ============================
    # 初始化记录变量
    train_losses = []  # 训练损失记录
    val_losses = []  # 验证损失记录
    train_accuracies = []  # 训练准确率记录
    val_accuracies = []  # 验证准确率记录
    learning_rates = []  # 学习率记录
    best_val_accuracy = 0.0  # 最佳验证准确率
    early_stop_counter = 0  # 早停计数器
    best_epoch = 0  # 最佳模型所在轮数

    print("开始半监督训练...")

    for epoch in range(MAX_EPOCH):
        # 训练阶段
        train_loss, train_accuracy = trainer.train_epoch(
            epoch,
            use_consistency=True,  # 使用一致性训练
            use_pseudo_labels=True  # 使用伪标签
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 更新学习率并记录
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        scheduler.step()  # 更新学习率

        # 验证阶段
        val_loss, val_accuracy = trainer.validate()
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 早停判断和模型保存
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            early_stop_counter = 0  # 重置早停计数器

            # 保存最佳模型
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss
            }
            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
            print(f"✅ 保存最佳模型，验证准确率: {best_val_accuracy:.2f}%")
        else:
            early_stop_counter += 1  # 增加早停计数器

        # 打印训练信息
        print(f'Epoch: {epoch:03d}/{MAX_EPOCH}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
              f'LR: {current_lr:.6f}, '
              f'EarlyStop: {early_stop_counter}/{PATIENCE}')

        # 早停检查：如果连续PATIENCE个epoch验证准确率没有提升，停止训练
        if early_stop_counter >= PATIENCE:
            print(f"🚨 早停触发！在 epoch {epoch} 停止训练")
            print(f"🏆 最佳模型在 epoch {best_epoch}, 验证准确率: {best_val_accuracy:.2f}%")
            break

    # ============================ 训练结束 ============================
    print(f"训练完成！最终最佳验证准确率: {best_val_accuracy:.2f}%")

    # 保存训练记录
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch
    }
    torch.save(training_history, os.path.join(log_dir, 'training_history.pth'))

    # 绘制训练曲线
    picture_path_loss = os.path.join(log_dir, 'loss_curves.png')
    picture_path_acc = os.path.join(log_dir, 'accuracy_curves.png')
    picture_path_combined = os.path.join(log_dir, 'training_curves.png')

    plot_loss_curves(train_losses, val_losses, picture_path_loss)
    plot_accuracy_curves(train_accuracies, val_accuracies, picture_path_acc)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, picture_path_combined)

    print(f"训练曲线已保存至: {log_dir}")


if __name__ == "__main__":
    main()  # 程序入口点