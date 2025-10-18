import numpy as np
import torch
import sys
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
sys.path.append('../tools')
import pandas as pd



def plot_loss_curves(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")

def plot_accuracy_curves(train_accuracies, val_accuracies, save_path=None):
    """
    绘制准确率曲线

    Args:
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)

    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 设置y轴范围从0开始
    plt.ylim(bottom=0)

    # 添加最佳准确率标注
    best_val_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_val_acc) + 1
    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    plt.text(best_epoch, best_val_acc / 2, f'Best: {best_val_acc:.2f}%\nEpoch: {best_epoch}',
             ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    绘制综合训练曲线（损失和准确率在一起）

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        save_path: 图片保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 添加最佳准确率标注
    best_val_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_val_acc) + 1
    ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax2.text(best_epoch, best_val_acc / 2, f'Best: {best_val_acc:.2f}%',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()