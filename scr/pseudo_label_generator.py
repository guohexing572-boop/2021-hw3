import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import DatasetFolder
from PIL import Image
import torchvision.transforms as transforms


class PseudoLabelGenerator:
    """伪标签生成器"""

    def __init__(self, model, unlabeled_loader, device, threshold=0.9):
        self.model = model
        self.unlabeled_loader = unlabeled_loader
        self.device = device
        self.threshold = threshold

    def generate_pseudo_dataset(self, confidence_threshold=None):
        """生成伪标签数据集"""
        if confidence_threshold is None:
            confidence_threshold = self.threshold

        self.model.eval()
        all_pseudo_data = []
        all_pseudo_labels = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.unlabeled_loader):
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predictions = torch.max(probabilities, 1)

                # 筛选高置信度样本
                mask = max_probs > confidence_threshold
                high_conf_data = data[mask].cpu()
                high_conf_preds = predictions[mask].cpu()

                if len(high_conf_data) > 0:
                    all_pseudo_data.append(high_conf_data)
                    all_pseudo_labels.append(high_conf_preds)

                if batch_idx % 10 == 0:
                    print(f"处理批次 {batch_idx}, 当前伪标签数量: {len(high_conf_data)}")

        if all_pseudo_data:
            pseudo_data = torch.cat(all_pseudo_data)
            pseudo_labels = torch.cat(all_pseudo_labels)
            pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)

            print(f"总共生成了 {len(pseudo_dataset)} 个伪标签样本")
            return pseudo_dataset
        else:
            print("未生成任何伪标签样本")
            return None


# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    from tools.hw3_model import resnet18_512  # 根据你的实际导入路径调整

    model = resnet18_512(num_classes=11)
    model.to(device)

    # 设置数据路径 - 这里需要你修改为实际路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(BASE_DIR)

    # 数据增强
    train_tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 定义图片加载函数
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    try:
        # 创建无标签数据集
        unlabeled_set = DatasetFolder(
            os.path.join(parent_dir, "food-11", "training", "unlabeled"),
            loader=pil_loader,
            extensions="jpg",
            transform=train_tfm
        )

        # 创建无标签数据加载器
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=32,
            shuffle=False,  # 生成伪标签时不需要shuffle
            num_workers=2,
            pin_memory=True
        )

        print(f"无标签数据集大小: {len(unlabeled_set)}")

        # 创建生成器
        generator = PseudoLabelGenerator(model, unlabeled_loader, device)

        # 生成伪标签数据集
        print("开始生成伪标签...")
        pseudo_dataset = generator.generate_pseudo_dataset(threshold=0.8)

        if pseudo_dataset is not None:
            # 保存伪标签数据集
            torch.save(pseudo_dataset, "pseudo_labeled_dataset.pth")
            print("伪标签数据集已保存为 'pseudo_labeled_dataset.pth'")

    except FileNotFoundError as e:
        print(f"数据路径错误: {e}")
        print("请检查以下路径是否存在:")
        print(os.path.join(parent_dir, "food-11", "training", "unlabeled"))
        print("\n如果路径不正确，请修改为你的实际数据路径")