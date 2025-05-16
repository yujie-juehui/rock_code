import torchmetrics
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score,
    ConfusionMatrix, AUROC
)
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
import pandas as pd
import torchvision
from torch.cuda.amp import GradScaler, autocast
import warnings
from torch.optim.lr_scheduler import _LRScheduler
import seaborn as sns  # 导入seaborn库

# 全局配置
# 在类定义前的全局配置部分替换为以下代码
# 替换原来的plt.rcParams配置为以下内容
plt.rcParams.update({
    'axes.unicode_minus': False,  # 解决负号显示问题
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'],  # Windows系统优先使用雅黑
    'axes.axisbelow': True,
})
# plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP兼容
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 用于更好的CUDA错误调试
warnings.filterwarnings('ignore', category=UserWarning)  # 过滤不必要的警告


def top_k_accuracy(output, target, k=3):
    """计算 Top-K 准确率"""
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / target.size(0)).item()

class RockImageClassifier:
    def __init__(self):
        # 配置参数
        self.config = {
            'image_size': 224,
            'batch_size': 64,
            'num_workers': min(4, os.cpu_count()),
            'num_epochs':55,
            'learning_rate': 1e-4,
            # 'momentum': 0.9,
            # 'step_size': 200,#衰减步数
            # 'gamma': 0.90,#衰减率
            'checkpoint_path': 'best_rock_classifier_resnet152.pth',
            'use_amp': True,
            'grad_clip': 1.0,
            'unfreeze_layers': 4,
            'weight_decay': 1e-5,
            'dropout_prob': 0.5,
            't_0': 16,
            'eta_min': 1e-5,
            't_mult': 2,
            'warmup_epochs':5 #学习率预热
        }

        # 设备设置
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")

        # 数据相关
        self.image_datasets = None
        self.dataloaders = None
        self.class_names = None
        self.dataset_sizes = None

        # 训练记录
        self.history = {
            'epoch': [],
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }

        # 模型
        self.model = None
        self.scaler = GradScaler(enabled=self.config['use_amp'])  # 混合精度梯度缩放器

    def _init_device(self) -> torch.device:
        """初始化计算设备"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """获取数据增强转换"""
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.config['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return {'train': train_transform, 'val': val_transform}

    def load_data(self, data_dir: str = '岩石图像-_split/dataset') -> None:
        """加载数据集"""
        try:
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"数据目录不存在: {os.path.abspath(data_dir)}")

            transforms = self._get_transforms()
            self.image_datasets = {
                phase: datasets.ImageFolder(os.path.join(data_dir, phase), transforms[phase])
                for phase in ['train', 'val']
            }

            self.dataloaders = {
                phase: torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=(phase == 'train'),
                    num_workers=self.config['num_workers'],
                    pin_memory=True,
                    persistent_workers=True if self.config['num_workers'] > 0 else False
                )
                for phase, dataset in self.image_datasets.items()
            }

            self.dataset_sizes = {phase: len(dataset) for phase, dataset in self.image_datasets.items()}
            self.class_names = self.image_datasets['train'].classes

            print(f"数据加载完成 - 训练集: {self.dataset_sizes['train']}, 验证集: {self.dataset_sizes['val']}")
            print(f"类别: {', '.join(self.class_names)}")

        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise

    def show_samples(self, num_images: int = 8) -> None:
        """显示样本图像"""
        if not self.dataloaders:
            raise ValueError("请先加载数据")

        images, labels = next(iter(self.dataloaders['train']))
        grid = torchvision.utils.make_grid(images[:num_images], nrow=4, padding=2)

        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        grid = grid * std + mean
        grid = torch.clamp(grid, 0, 1)

        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0))
        if hasattr(self, 'class_names'):
            class_labels = [self.class_names[c] for c in labels[:num_images]]
            plt.title(f'训练样本示例\n类别: {class_labels}', fontsize=12)
        else:
            plt.title(f'样本图像 (共{num_images}张)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

    def init_model(self) -> None:
        """初始化模型"""
        if not self.class_names:
            raise ValueError("请先加载数据")

        # 加载预训练模型
        self.model = models.resnet152(weights='IMAGENET1K_V2')
        # self.model = models.resnet152(weights=None)

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻部分层
        unfreeze_layers = self.config['unfreeze_layers']
        if unfreeze_layers > 0:
            children = list(self.model.children())
            for layer in children[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"解冻最后{unfreeze_layers}层进行微调")

        # 修改分类头
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(self.config['dropout_prob']),
            nn.Linear(num_features, len(self.class_names))
        )

        # 梯度检查点 (PyTorch 2.0+)
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)
            print("启用梯度检查点以节省显存")

        self.model = self.model.to(self.device)

        # 打印可训练参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型初始化完成，总参数: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"分类数: {len(self.class_names)}")

    def train_epoch(self, phase: str, epoch: int, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[
        float, float]:
        """训练或验证一个epoch"""
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            with torch.set_grad_enabled(phase == 'train'), torch.amp.autocast(device_type='cuda',
                                                                              enabled=self.config['use_amp']):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                #添加显存监控
                if phase == 'train' and self.device.type == 'cuda':
                    print(f"当前显存占用: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            if phase == 'train':
                self.scaler.scale(loss).backward()
                if self.config['grad_clip'] > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(optimizer)
                self.scaler.update()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            processed_samples += batch_size

            # 打印进度信息
            if processed_samples % (10 * self.config['batch_size']) == 0:
                print(f'{phase.capitalize()} Epoch: {epoch} [{processed_samples}/{self.dataset_sizes[phase]}]')

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        return epoch_loss, epoch_acc.item()

    # 使用方式

    def train_model(self) -> None:
        """训练模型"""
        if not all([self.model, self.dataloaders]):
            raise ValueError("模型或数据未初始化")

        # 训练设置
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-4
        )

        # 学习率调度器 (预热+余弦退火)
        def warmup_lr_scheduler(epoch):
            return min(1.0, (epoch + 1) / self.config['warmup_epochs'])

        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler),
                lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.config['t_0'],
                    T_mult=self.config['t_mult'],
                    eta_min=self.config['eta_min']
                )
            ],
            milestones=[self.config['warmup_epochs']]
        )

        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.config['t_0'],  # 初始周期长度（步数或epoch数）
        #     T_mult=self.config.get('t_mult', 2),  # 周期长度乘数（默认1，即固定周期）
        #     eta_min=self.config['eta_min']  # 学习率最小值
        # )

        best_acc = 0.0
        start_time = time.time()

        print("\n开始训练...")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")

        for epoch in range(self.config['num_epochs']):
            epoch_time = time.time()
            self.history['epoch'].append(epoch + 1)

            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 20)

            for phase in ['train', 'val']:
                epoch_loss, epoch_acc = self.train_epoch(phase, epoch + 1, optimizer, criterion)
                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc)

                print(f"{phase.capitalize()} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # 验证阶段处理
                if phase == 'val':
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    self.history['lr'].append(current_lr)
                    print(f'Learning Rate: {current_lr:.2e}')

                    # 保存最佳模型
                    # if epoch_acc > best_acc:
                    #     best_acc = epoch_acc
                    #     torch.save({
                    #         'epoch': epoch + 1,
                    #         'state_dict': self.model.state_dict(),
                    #         'optimizer': optimizer.state_dict(),
                    #         'accuracy': best_acc,
                    #         'classes': self.class_names
                    #     }, self.config['checkpoint_path'])
                    #     print(f"新最佳准确率: {best_acc:.4f}")
                    # 保存最佳模型（仅保存模型参数）
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # 仅保存模型参数（state_dict）和类别名称
                        torch.save({
                            'state_dict': self.model.state_dict(),
                            'classes': self.class_names  # 可选：保存类别名称
                        }, self.config['checkpoint_path'])
                        print(f"新最佳准确率: {best_acc:.4f}")

            # 保存指标
        self._save_metrics()
        print(f"Epoch 耗时: {time.time() - epoch_time:.2f}s")

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 耗时: {total_time // 60:.0f}m {total_time % 60:.2f}s")
        print(f"最佳验证准确率: {best_acc:.4f}")

        # 加载最佳模型
        self._load_best_model()
        self.plot_history()

    def _save_metrics(self) -> None:
        """保存训练指标"""
        import datetime
        df = pd.DataFrame({
            'epoch': self.history['epoch'],
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_acc': self.history['train_acc'],
            'val_acc': self.history['val_acc'],
            'learning_rate': self.history['lr']
        })

        # 添加时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'training_metrics_resnet152_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"训练指标已保存到 {csv_path}")

    # def _load_best_model(self) -> None:
    #     """加载最佳模型"""
    #     checkpoint = torch.load(self.config['checkpoint_path'])
    #     self.model.load_state_dict(checkpoint['state_dict'])
    #     print(f"已加载最佳模型 (Epoch {checkpoint['epoch']}, Acc: {checkpoint['accuracy']:.4f})")
    def _load_best_model(self) -> None:
        """加载最佳模型（仅参数版本）"""
        checkpoint = torch.load(self.config['checkpoint_path'])
        self.model.load_state_dict(checkpoint['state_dict'])
        # 如果保存了类别信息，可以在这里恢复
        if 'classes' in checkpoint:
            self.class_names = checkpoint['classes']
        print("已加载最佳模型参数")

    def plot_history(self) -> None:
        """绘制训练曲线"""
        plt.figure(figsize=(18, 6))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='训练损失')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='验证损失')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练和验证损失曲线', fontsize=14)
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.history['epoch'], self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['epoch'], self.history['val_acc'], label='验证准确率')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('训练和验证准确率曲线', fontsize=14)
        plt.legend()

        # 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.history['epoch'], self.history['lr'], label='学习率')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('学习率变化曲线', fontsize=14)
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history_resnet152.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        # 初始化指标计算器
        metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=len(self.class_names)).to(self.device),
            'precision': Precision(task='multiclass', num_classes=len(self.class_names), average='macro').to(
                self.device),
            'recall': Recall(task='multiclass', num_classes=len(self.class_names), average='macro').to(self.device),
            'f1': F1Score(task='multiclass', num_classes=len(self.class_names), average='macro').to(self.device),
            'auroc': AUROC(task='multiclass', num_classes=len(self.class_names), average='macro').to(self.device),
            'confusion_matrix': ConfusionMatrix(task='multiclass', num_classes=len(self.class_names)).to(self.device)
        }

        # 为每个类别单独记录指标（可选）
        class_metrics = {}
        for i, name in enumerate(self.class_names):
            class_metrics[f'precision_{name}'] = Precision(task='multiclass', num_classes=len(self.class_names),
                                                           average='none').to(self.device)
            class_metrics[f'recall_{name}'] = Recall(task='multiclass', num_classes=len(self.class_names),
                                                     average='none').to(self.device)

        softmax = nn.Softmax(dim=1)
        top_k_accuracies = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['val'], desc="评估中"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=self.config['use_amp']):
                    outputs = self.model(inputs)

                outputs_probs = softmax(outputs)

                # 更新所有指标
                for metric in metrics.values():
                    if isinstance(metric, AUROC):
                        metric.update(outputs_probs, labels)
                    else:
                        metric.update(outputs, labels)

                # 更新每个类别的指标（可选）
                for metric in class_metrics.values():
                    metric.update(outputs, labels)

                # 计算 Top-K 准确率
                top_k_acc = top_k_accuracy(outputs, labels, k=3)
                top_k_accuracies.append(top_k_acc)

        # 计算最终指标
        # results = {name: metric.compute().item() for name, metric in metrics.items()}
        # results['top3_accuracy'] = sum(top_k_accuracies) / len(top_k_accuracies)
        scalar_metrics = {name: metric.compute().item() for name, metric in metrics.items() if
                          name != 'confusion_matrix'}
        results = scalar_metrics
        results['confusion_matrix'] = metrics['confusion_matrix'].compute()
        results['top3_accuracy'] = sum(top_k_accuracies) / len(top_k_accuracies)

        # 计算每个类别的指标（可选）
        for name, metric in class_metrics.items():
            class_results = metric.compute()
            for i, cls_name in enumerate(self.class_names):
                results[f"{name.split('_')[0]}_{cls_name}"] = class_results[i].item()

        # 打印结果
        print("\n=== 评估结果 ===")
        print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
        print(f"精确率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1分数 (F1): {results['f1']:.4f}")
        print(f"Top-3准确率: {results['top3_accuracy']:.4f}")
        print(f"AUC-ROC: {results['auroc']:.4f}")

        # 可视化混淆矩阵
        cm = results['confusion_matrix'].cpu().numpy()
        #归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return results

    def visualize_results(self, num_images: int = 12) -> None:
        """可视化预测结果"""
        self.model.eval()
        fig, axes = plt.subplots(nrows=(num_images + 3) // 4, ncols=4, figsize=(16, num_images))
        axes = axes.flatten()

        images_processed = 0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                with torch.amp.autocast(device_type='cuda', enabled=self.config['use_amp']):
                    outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size(0)):
                    if images_processed >= num_images:
                        break

                    ax = axes[images_processed]
                    img = inputs.cpu().data[j]

                    # 反归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)

                    ax.imshow(img.permute(1, 2, 0))
                    title_color = 'green' if preds[j] == labels[j] else 'red'
                    ax.set_title(f'预测: {self.class_names[preds[j]]}\n真实: {self.class_names[labels[j]]}',
                                 color=title_color, fontsize=9)
                    ax.axis('off')

                    images_processed += 1

                if images_processed >= num_images:
                    break

        plt.tight_layout()
        plt.savefig('predictions_resnet152.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run(self, data_dir: str = 'rock-images-ori') -> None:
        """运行完整流程"""
        try:
            print("\n=== 岩石图像分类 ===")

            # 1. 加载数据
            print("\n[1/5] 加载数据...")
            self.load_data(data_dir)

            # 2. 显示样本
            print("\n[2/5] 显示样本...")
            self.show_samples()

            # 3. 初始化模型
            print("\n[3/5] 初始化模型...")
            self.init_model()

            # 4. 训练模型
            print("\n[4/5] 训练模型...")
            self.train_model()

            # 5. 评估和可视化
            print("\n[5/5] 评估模型...")
            self.evaluate()
            self.visualize_results()

            print("\n=== 完成! ===")

        except Exception as e:
            print(f"\n错误: {str(e)}")
            raise


if __name__ == '__main__':
    plt.ion()  # 交互模式
    try:
        classifier = RockImageClassifier()
        classifier.run()
    finally:
        plt.ioff()
