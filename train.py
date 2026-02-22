import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

import wandb
wandb.login(key="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")
# -------------------------
# Dataset
# -------------------------

def get_pacs_dataset(root, domain, transform):
    domain_path = os.path.join(root, domain)
    return ImageFolder(domain_path, transform=transform)

# -------------------------
# Utils
# -------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_grad_norm(model):
    total_norm = 0.0
    layer_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            layer_norms[name] = param_norm
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm, layer_norms

def accuracy(outputs, targets):
    _, pred = outputs.max(1)
    correct = pred.eq(targets).sum().item()
    return correct / targets.size(0)

# -------------------------
# Mixup
# -------------------------

def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------
# Hybrid Norm (BN + IN 혼합)
# -------------------------

class HybridNorm1d(nn.Module):
    """
    Feature vector(1D)에 대해 BN과 IN을 혼합한 Norm.
    - BN: 배치 전체의 통계를 활용 → 도메인 공유 정보 정규화
    - IN: 샘플별 통계를 활용 → 도메인 특화 스타일 제거
    - 학습 가능한 가중치 alpha로 두 norm을 혼합
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        # IN for 1D: 각 샘플을 독립적으로 정규화 (LayerNorm과 동일한 효과)
        self.in_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 학습 가능한 혼합 비율

    def forward(self, x):
        # alpha를 [0, 1]로 클램핑
        alpha = self.alpha.clamp(0.0, 1.0)
        return alpha * self.bn(x) + (1 - alpha) * self.in_norm(x)

# -------------------------
# Early Stopping
# -------------------------

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# -------------------------
# Style Augmentation
# -------------------------

class StyleAugment2d(nn.Module):
    def __init__(self, p=0.5, scale=0.3, shift=0.3):
        super().__init__()
        self.p = p
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        if not self.training:
            return x
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.size()

        mean = x.mean(dim=[2,3], keepdim=True)
        std  = x.std(dim=[2,3], keepdim=True) + 1e-6

        normalized = (x - mean) / std

        noise_scale = torch.randn(B, C, 1, 1, device=x.device) * self.scale + 1.0
        noise_shift = torch.randn(B, C, 1, 1, device=x.device) * self.shift

        return normalized * noise_scale + noise_shift

# -------------------------
# Model
# -------------------------

class ResNet18WithStyleAug(nn.Module):
    def __init__(self, num_classes=7, use_pre_fc_norm=False,
                 use_style_aug=False,
                 style_p=0.5,
                 style_scale=0.3,
                 style_shift=0.3,
                 style_position="layer2"):

        super().__init__()

        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.use_pre_fc_norm = use_pre_fc_norm
        if use_pre_fc_norm:
            self.pre_fc_norm = HybridNorm1d(in_features)

        self.use_style_aug = use_style_aug
        self.style_position = style_position

        if use_style_aug:
            self.style_aug = StyleAugment2d(style_p, style_scale, style_shift)

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
      x = self.backbone.conv1(x)
      x = self.backbone.bn1(x)
      x = self.backbone.relu(x)
      x = self.backbone.maxpool(x)

      x = self.backbone.layer1(x)
      if self.use_style_aug and self.style_position == "layer1":
          x = self.style_aug(x)

      x = self.backbone.layer2(x)
      if self.use_style_aug and self.style_position == "layer2":
          x = self.style_aug(x)

      x = self.backbone.layer3(x)
      if self.use_style_aug and self.style_position == "layer3":
          x = self.style_aug(x)

      x = self.backbone.layer4(x)

      x = self.backbone.avgpool(x)
      x = torch.flatten(x, 1)

      if self.use_pre_fc_norm:
          x = self.pre_fc_norm(x)

      return self.classifier(x)

# -------------------------
# Main
# -------------------------

def main(args):
    args.train_domain = "photo"
    args.test_domain  = "sketch"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Transform
    # -------------------------

    train_aug = []
    
    if args.random_crop > 0:
        train_aug.append(transforms.RandomResizedCrop(args.random_crop))
    else:
        train_aug.append(transforms.Resize(224))
        train_aug.append(transforms.CenterCrop(224))
        
    if args.hflip_prob > 0:
        train_aug.append(transforms.RandomHorizontalFlip(p=args.hflip_prob))

    if args.grayscale > 0:
        train_aug.append(transforms.RandomGrayscale(p=args.grayscale))

    if any([args.cj_brightness, args.cj_contrast,
            args.cj_saturation, args.cj_hue]):
        train_aug.append(
            transforms.ColorJitter(
                brightness=args.cj_brightness,
                contrast=args.cj_contrast,
                saturation=args.cj_saturation,
                hue=args.cj_hue
            )
        )

    if args.blur_kernel >= 0.1:
        train_aug.append(transforms.GaussianBlur(args.blur_kernel))
    train_aug.append(transforms.ToTensor())
    train_transform = transforms.Compose(train_aug)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    train_set = get_pacs_dataset(args.data_root, args.train_domain, train_transform)
    test_set  = get_pacs_dataset(args.data_root, args.test_domain, test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ResNet18WithStyleAug(
        num_classes=len(train_set.classes),
        use_pre_fc_norm=args.use_norm,
        use_style_aug=args.use_style_aug,
        style_p=args.style_p,
        style_scale=args.style_scale,
        style_shift=args.style_shift,
        style_position=args.style_position
    ).to(device)

    # -------------------------
    # Optimizer / Scheduler
    # -------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # -------------------------
    # Early Stopping
    # -------------------------

    early_stopper = EarlyStopping(patience=args.patience)

    # -------------------------
    # Wandb
    # -------------------------

    wandb.init(
        project=args.project,
        name=args.run_name,
        id=args.run_name,
        resume="allow",
        config=vars(args)
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{wandb.run.name}.pt")

    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        # EarlyStopping 상태 복원
        if "early_stopper" in ckpt:
            early_stopper.best    = ckpt["early_stopper"]["best"]
            early_stopper.counter = ckpt["early_stopper"]["counter"]

    # -------------------------
    # Train Loop
    # -------------------------

    for epoch in range(start_epoch, args.epochs):

        model.train()

        # BN freeze 옵션: hybrid norm을 쓰지 않을 때 backbone BN을 고정
        if not args.use_norm:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        train_loss        = 0.0
        train_acc         = 0.0
        grad_norm_total_epoch = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if args.use_mixup:
                images, y_a, y_b, lam = mixup_data(images, labels, args.mixup_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("NaN detected. Stopping.")
                return

            loss.backward()

            grad_norm_total, _ = compute_grad_norm(model)
            grad_norm_total_epoch += grad_norm_total

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc  += accuracy(outputs, labels)  # 원래 label 기준

        train_loss            /= len(train_loader)
        train_acc             /= len(train_loader)
        grad_norm_total_epoch /= len(train_loader)

        # -------------------------
        # 5 epoch마다 Test
        # -------------------------
        test_acc = 0.0
        if(epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    test_acc += accuracy(outputs, labels)

            test_acc /= len(test_loader)

        # HybridNorm의 학습된 alpha 값도 함께 로깅
        hybrid_alpha = None
        if args.use_norm:
            hybrid_alpha = model.pre_fc_norm.alpha.item()

        log_dict = {
            "epoch":            epoch,
            "train_loss":       train_loss,
            "train_acc":        train_acc,
            "test_acc":         test_acc,
            "grad_norm_total":  grad_norm_total_epoch,
            "lr":               scheduler.get_last_lr()[0],
        }
        if hybrid_alpha is not None:
            log_dict["hybrid_norm_alpha"] = hybrid_alpha

        wandb.log(log_dict)

        # -------------------------
        # Checkpoint
        # -------------------------

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopper": {
                "best":    early_stopper.best,
                "counter": early_stopper.counter,
            },
        }, ckpt_path)

        alpha_str = f" | HybridAlpha: {hybrid_alpha:.4f}" if hybrid_alpha is not None else ""
        print(f"[Epoch {epoch}] Train: {train_acc:.4f} | Test: {test_acc:.4f} | GradNorm: {grad_norm_total_epoch:.4f}{alpha_str}")

        scheduler.step()

        # -------------------------
        # Early Stopping
        # -------------------------

        early_stopper.step(test_acc)
        if early_stopper.stop:
            print(f"Early stopping triggered at epoch {epoch} (best test_acc: {early_stopper.best:.4f})")
            break

    wandb.finish()

# -------------------------
# Argparse
# -------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="ood-style-generalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name",   type=str, default=None)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing",type=float, default=0.0)
    # Norm 옵션
    parser.add_argument("--use_norm", action="store_true",
                        help="FC 직전에 BN+IN 혼합 HybridNorm 적용")
    # Mixup 옵션
    parser.add_argument("--use_mixup",   action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    # Early Stopping
    parser.add_argument("--patience",    type=int,   default=15,
                        help="EarlyStopping patience (epochs)")
    # Augmentation
    parser.add_argument("--random_crop",   type=float,   default=224)
    parser.add_argument("--hflip_prob",    type=float, default=0.5)
    parser.add_argument("--grayscale",     type=float, default=0.0)
    parser.add_argument("--cj_brightness", type=float, default=0.2)
    parser.add_argument("--cj_contrast",   type=float, default=0.2)
    parser.add_argument("--cj_saturation", type=float, default=0.2)
    parser.add_argument("--cj_hue",        type=float, default=0.1)
    parser.add_argument("--blur_kernel",   type=int,   default=3)
    # Style Augmentation
    parser.add_argument("--use_style_aug", action="store_true")
    parser.add_argument("--style_p", type=float, default=0.5)
    parser.add_argument("--style_scale", type=float, default=0.3)
    parser.add_argument("--style_shift", type=float, default=0.3)
    parser.add_argument("--style_position", type=str, default="layer2",
                        choices=["layer1", "layer2", "layer3"])
    # Path
    parser.add_argument("--ckpt_dir",      type=str, default="./ckpt")
    
    parser.add_argument("--data_root", type=str, default="./PACS")

    args = parser.parse_args()
    main(args)
