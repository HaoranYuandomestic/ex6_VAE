import os
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
except Exception:
    HAS_FID = False


# ============================================================
# 配置
# ============================================================
class Config:
    # 数据路径
    photo_dir = r"./val2014"
    art_dir = r"./wikiart_subset"

    # 作业要求
    photo_count = 10000
    art_count = 8000
    image_size = 128
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 训练
    seed = 42
    batch_size = 8              # RTX8000 48G 上 256x256 通常可跑；不稳就改成 8
    num_workers = 0
    lr = 2e-4
    min_lr = 1e-5
    weight_decay = 1e-5
    epochs = 50
    patience = 12
    grad_clip = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()  # 自动使用混合精度，节省显存并加速训练

    # CNN-VAE
    latent_dim = 256
    domain_emb_dim = 64
    beta_max = 5e-4              # 关键：明显低于你原来的 0.01，避免直接塌成均值图
    kl_warmup_epochs = 20
    free_bits = 0.02             # 防止后验完全塌缩

    # 输出
    vis_dir = "visualizations"
    ckpt_path = "best_cnn_cvae.pth"
    fid_num_samples = 500


cfg = Config()


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def list_image_paths(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)



def select_and_split_paths(folder, max_count, seed=42):
    paths = list_image_paths(folder)
    if len(paths) < max_count:
        raise ValueError(f"{folder} 中图片不足 {max_count} 张，当前只有 {len(paths)} 张。")

    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:max_count]

    n = len(paths)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test = n - n_train - n_val

    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]
    return train_paths, val_paths, test_paths



def denorm_img(x):
    # [-1, 1] -> [0, 1]
    x = (x * 0.5 + 0.5)
    x = np.clip(x, 0.0, 1.0)
    return np.transpose(x, (1, 2, 0))



def save_grid(images, save_path, titles=None, nrow=6, figsize_scale=2.2):
    n = len(images)
    ncol = min(nrow, n)
    nrow_actual = math.ceil(n / ncol)

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(figsize_scale * ncol, figsize_scale * nrow_actual))
    if nrow_actual == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    idx = 0
    for r in range(nrow_actual):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if idx < n:
                ax.imshow(denorm_img(images[idx]))
                if titles is not None:
                    ax.set_title(titles[idx], fontsize=9)
            idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 数据集
# ============================================================
class ImageDomainDataset(Dataset):
    def __init__(self, paths, domain_label, train=False):
        self.paths = paths
        self.domain_label = domain_label

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(int(cfg.image_size * 1.10)),
                transforms.RandomCrop(cfg.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.domain_label



def build_dataloaders():
    p_train, p_val, p_test = select_and_split_paths(cfg.photo_dir, cfg.photo_count, seed=cfg.seed)
    a_train, a_val, a_test = select_and_split_paths(cfg.art_dir, cfg.art_count, seed=cfg.seed + 1)

    photo_train = ImageDomainDataset(p_train, 0, train=True)
    photo_val = ImageDomainDataset(p_val, 0, train=False)
    photo_test = ImageDomainDataset(p_test, 0, train=False)

    art_train = ImageDomainDataset(a_train, 1, train=True)
    art_val = ImageDomainDataset(a_val, 1, train=False)
    art_test = ImageDomainDataset(a_test, 1, train=False)

    train_dataset = ConcatDataset([photo_train, art_train])
    val_dataset = ConcatDataset([photo_val, art_val])
    test_dataset = ConcatDataset([photo_test, art_test])

    # 平衡采样，避免 10000 vs 8000 导致域偏移
    weights = [1.0 / len(photo_train)] * len(photo_train) + [1.0 / len(art_train)] * len(art_train)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    photo_test_loader = DataLoader(
        photo_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    art_test_loader = DataLoader(
        art_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    print(f"Photo total: {cfg.photo_count} -> train/val/test = {len(p_train)}/{len(p_val)}/{len(p_test)}")
    print(f"Art total:   {cfg.art_count} -> train/val/test = {len(a_train)}/{len(a_val)}/{len(a_test)}")
    print(f"Train total: {len(train_dataset)}, Val total: {len(val_dataset)}, Test total: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, photo_test_loader, art_test_loader


# ============================================================
# CNN-CVAE 模型
# ============================================================
class ResDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g1 = max(1, min(16, in_ch))
        g2 = max(1, min(16, out_ch))
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class FiLM(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(max(1, min(16, ch)), ch, affine=False)
        self.to_scale_shift = nn.Linear(emb_dim, ch * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return h * (1.0 + scale) + shift


class ResUp(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film1 = FiLM(out_ch, emb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film2 = FiLM(out_ch, emb_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond):
        x_up = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip(x_up)
        h = self.conv1(x_up)
        h = self.act(self.film1(h, cond))
        h = self.conv2(h)
        h = self.act(self.film2(h, cond))
        return h + skip


class CNNCVAE(nn.Module):
    def __init__(self, latent_dim=256, domain_emb_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.domain_emb = nn.Embedding(2, domain_emb_dim)

        self.stem = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.enc1 = ResDown(64, 64)      # 128
        self.enc2 = ResDown(64, 128)     # 64
        self.enc3 = ResDown(128, 256)    # 32
        self.enc4 = ResDown(256, 384)    # 16
        self.enc5 = ResDown(384, 512)    # 8

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.to_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.to_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.dec1 = ResUp(512, 384, domain_emb_dim)  # 16
        self.dec2 = ResUp(384, 256, domain_emb_dim)  # 32
        self.dec3 = ResUp(256, 128, domain_emb_dim)  # 64
        self.dec4 = ResUp(128, 64, domain_emb_dim)   # 128
        self.dec5 = ResUp(64, 32, domain_emb_dim)    # 256
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.stem(x)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.pool(h)
        h = h.flatten(1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, domain_ids):
        cond = self.domain_emb(domain_ids)
        h = self.fc(z).view(-1, 512, 4, 4)
        h = self.dec1(h, cond)
        h = self.dec2(h, cond)
        h = self.dec3(h, cond)
        h = self.dec4(h, cond)
        h = self.dec5(h, cond)
        return self.out(h)

    def forward(self, x, domain_ids):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, domain_ids)
        return recon, mu, logvar

    @torch.no_grad()
    def translate(self, x, target_domain_ids, use_mean=True):
        mu, logvar = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        out = self.decode(z, target_domain_ids)
        return out, mu, logvar


# ============================================================
# 损失与评估
# ============================================================
def vae_loss(recon, x, mu, logvar, beta, free_bits=0.02):
    # 混合重建损失：比单纯 MSE 清晰，比单纯 L1 稳定
    l1 = F.l1_loss(recon, x, reduction="mean")
    mse = F.mse_loss(recon, x, reduction="mean")
    recon_loss = 0.8 * l1 + 0.2 * mse

    # KL with free bits，减少 posterior collapse
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()

    total = recon_loss + beta * kl
    return total, recon_loss, kl


@torch.no_grad()
def evaluate(model, loader, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mse = 0.0
    count = 0

    use_amp = cfg.use_amp and cfg.device.startswith("cuda")
    for x, domain in loader:
        x = x.to(cfg.device, non_blocking=True)
        domain = domain.to(cfg.device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            recon, mu, logvar = model(x, domain)
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta, cfg.free_bits)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs
        total_mse += F.mse_loss(recon, x, reduction="mean").item() * bs
        count += bs

    return {
        "loss": total_loss / count,
        "recon": total_recon / count,
        "kl": total_kl / count,
        "mse": total_mse / count,
    }



def get_beta(epoch):
    # 前 3 个 epoch 基本不压 KL，先把重建学出来
    if epoch <= 3:
        return 1e-6
    progress = min(1.0, (epoch - 3) / max(1, cfg.kl_warmup_epochs - 3))
    return cfg.beta_max * progress



def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=cfg.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and cfg.device.startswith("cuda"))

    best_val = float("inf")
    wait = 0
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "val_mse": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        use_amp = cfg.use_amp and cfg.device.startswith("cuda")
        beta = get_beta(epoch)

        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        count = 0

        for x, domain in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            domain = domain.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                recon, mu, logvar = model(x, domain)
                loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta, cfg.free_bits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_recon += recon_loss.item() * bs
            running_kl += kl.item() * bs
            count += bs

        train_loss = running_loss / count
        train_recon = running_recon / count
        train_kl = running_kl / count

        val_stats = evaluate(model, val_loader, beta)
        scheduler.step(val_stats["loss"])

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])
        history["val_kl"].append(val_stats["kl"])
        history["val_mse"].append(val_stats["mse"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{cfg.epochs}] | lr={current_lr:.2e} | beta={beta:.6f} | "
            f"train_loss={train_loss:.5f} train_recon={train_recon:.5f} train_kl={train_kl:.5f} | "
            f"val_loss={val_stats['loss']:.5f} val_recon={val_stats['recon']:.5f} val_mse={val_stats['mse']:.5f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(model.state_dict(), cfg.ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return history


# ============================================================
# 可视化
# ============================================================
def plot_loss_curves(history):
    ensure_dir(cfg.vis_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_total")
    plt.plot(epochs, history["val_loss"], label="val_total")
    plt.plot(epochs, history["train_recon"], label="train_recon")
    plt.plot(epochs, history["val_recon"], label="val_recon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN-CVAE Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.vis_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_kl"], label="train_kl")
    plt.plot(epochs, history["val_kl"], label="val_kl")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.title("KL Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.vis_dir, "kl_curve.png"), dpi=150)
    plt.close()



def get_batch(loader):
    x, d = next(iter(loader))
    return x.to(cfg.device), d.to(cfg.device)


@torch.no_grad()
def save_reconstruction_figure(model, photo_loader, art_loader, n_show=6):
    ensure_dir(cfg.vis_dir)
    model.eval()

    x_p, d_p = get_batch(photo_loader)
    x_a, d_a = get_batch(art_loader)

    recon_p, _, _ = model(x_p[:n_show], d_p[:n_show])
    recon_a, _, _ = model(x_a[:n_show], d_a[:n_show])

    x_p = x_p[:n_show].cpu().numpy()
    x_a = x_a[:n_show].cpu().numpy()
    recon_p = recon_p.cpu().numpy()
    recon_a = recon_a.cpu().numpy()

    fig, axes = plt.subplots(4, n_show, figsize=(2.2 * n_show, 8))
    for i in range(n_show):
        axes[0, i].imshow(denorm_img(x_p[i]))
        axes[1, i].imshow(denorm_img(recon_p[i]))
        axes[2, i].imshow(denorm_img(x_a[i]))
        axes[3, i].imshow(denorm_img(recon_a[i]))
        for r in range(4):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Photo", fontsize=10)
    axes[1, 0].set_ylabel("Photo Recon", fontsize=10)
    axes[2, 0].set_ylabel("Art", fontsize=10)
    axes[3, 0].set_ylabel("Art Recon", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.vis_dir, "reconstruction_compare.png"), dpi=150)
    plt.close()


@torch.no_grad()
def save_translation_figure(model, photo_loader, art_loader, n_show=6):
    ensure_dir(cfg.vis_dir)
    model.eval()

    x_p, _ = get_batch(photo_loader)
    x_a, _ = get_batch(art_loader)

    x_p = x_p[:n_show]
    x_a = x_a[:n_show]
    target_art = torch.ones(x_p.size(0), dtype=torch.long, device=cfg.device)
    target_photo = torch.zeros(x_a.size(0), dtype=torch.long, device=cfg.device)

    p2a, _, _ = model.translate(x_p, target_art, use_mean=True)
    a2p, _, _ = model.translate(x_a, target_photo, use_mean=True)

    x_p = x_p.cpu().numpy()
    x_a = x_a.cpu().numpy()
    p2a = p2a.cpu().numpy()
    a2p = a2p.cpu().numpy()

    fig, axes = plt.subplots(4, n_show, figsize=(2.2 * n_show, 8))
    for i in range(n_show):
        axes[0, i].imshow(denorm_img(x_p[i]))
        axes[1, i].imshow(denorm_img(p2a[i]))
        axes[2, i].imshow(denorm_img(x_a[i]))
        axes[3, i].imshow(denorm_img(a2p[i]))
        for r in range(4):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Photo Input", fontsize=10)
    axes[1, 0].set_ylabel("Photo -> Art", fontsize=10)
    axes[2, 0].set_ylabel("Art Input", fontsize=10)
    axes[3, 0].set_ylabel("Art -> Photo", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.vis_dir, "style_transfer_bidirectional.png"), dpi=150)
    plt.close()


@torch.no_grad()
def save_interpolation_figure(model, photo_loader, art_loader):
    ensure_dir(cfg.vis_dir)
    model.eval()

    x_p, _ = get_batch(photo_loader)
    x_a, _ = get_batch(art_loader)
    x_p = x_p[:1]
    x_a = x_a[:1]

    mu_p, _ = model.encode(x_p)
    mu_a, _ = model.encode(x_a)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    target_art = torch.ones(1, dtype=torch.long, device=cfg.device)
    outputs = []
    for alpha in alphas:
        z = (1 - alpha) * mu_p + alpha * mu_a
        img = model.decode(z, target_art)
        outputs.append(img[0].cpu().numpy())

    images = [x_p[0].cpu().numpy()] + outputs + [x_a[0].cpu().numpy()]
    titles = ["photo"] + [f"a={a:.2f}" for a in alphas] + ["art"]
    save_grid(images, os.path.join(cfg.vis_dir, "latent_interpolation.png"), titles=titles, nrow=len(images))


@torch.no_grad()
def save_random_generation_figure(model, n_show=8):
    ensure_dir(cfg.vis_dir)
    model.eval()

    z = torch.randn(n_show, cfg.latent_dim, device=cfg.device)
    photo_domain = torch.zeros(n_show, dtype=torch.long, device=cfg.device)
    art_domain = torch.ones(n_show, dtype=torch.long, device=cfg.device)

    gen_photo = model.decode(z, photo_domain).cpu().numpy()
    gen_art = model.decode(z, art_domain).cpu().numpy()

    save_grid(list(gen_photo), os.path.join(cfg.vis_dir, "random_generation_photo.png"), nrow=4)
    save_grid(list(gen_art), os.path.join(cfg.vis_dir, "random_generation_art.png"), nrow=4)


# ============================================================
# FID
# ============================================================
def to_uint8_batch(x):
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return (x * 255.0).to(torch.uint8)


@torch.no_grad()
def compute_fid(model, source_loader, target_loader, target_domain, max_samples=500):
    if not HAS_FID:
        return None

    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(cfg.device)

    real_count = 0
    for x_real, _ in target_loader:
        x_real = x_real.to(cfg.device)
        fid.update(to_uint8_batch(x_real), real=True)
        real_count += x_real.size(0)
        if real_count >= max_samples:
            break

    fake_count = 0
    for x_src, _ in source_loader:
        x_src = x_src.to(cfg.device)
        domain_ids = torch.full((x_src.size(0),), target_domain, dtype=torch.long, device=cfg.device)
        x_fake, _, _ = model.translate(x_src, domain_ids, use_mean=True)
        fid.update(to_uint8_batch(x_fake), real=False)
        fake_count += x_fake.size(0)
        if fake_count >= max_samples:
            break

    return float(fid.compute().item())


# ============================================================
# 主程序
# ============================================================
def main():
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ensure_dir(cfg.vis_dir)

    train_loader, val_loader, test_loader, photo_test_loader, art_test_loader = build_dataloaders()
    model = CNNCVAE(cfg.latent_dim, cfg.domain_emb_dim).to(cfg.device)

    print(f"Device: {cfg.device}")
    print("Start training...")
    history = train(model, train_loader, val_loader)

    if os.path.exists(cfg.ckpt_path):
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location=cfg.device))
        print(f"Loaded best checkpoint: {cfg.ckpt_path}")

    beta_eval = cfg.beta_max
    test_stats = evaluate(model, test_loader, beta_eval)
    print("\n===== Test Reconstruction Metrics =====")
    print(f"Test total loss: {test_stats['loss']:.6f}")
    print(f"Test recon loss: {test_stats['recon']:.6f}")
    print(f"Test KL loss:    {test_stats['kl']:.6f}")
    print(f"Test MSE:        {test_stats['mse']:.6f}")

    plot_loss_curves(history)
    save_reconstruction_figure(model, photo_test_loader, art_test_loader, n_show=6)
    save_translation_figure(model, photo_test_loader, art_test_loader, n_show=6)
    save_interpolation_figure(model, photo_test_loader, art_test_loader)
    save_random_generation_figure(model, n_show=8)

    print("\n===== FID =====")
    if HAS_FID:
        try:
            fid_p2a = compute_fid(model, photo_test_loader, art_test_loader, 1, cfg.fid_num_samples)
            fid_a2p = compute_fid(model, art_test_loader, photo_test_loader, 0, cfg.fid_num_samples)
            print(f"FID Photo -> Art: {fid_p2a:.4f}")
            print(f"FID Art -> Photo: {fid_a2p:.4f}")
        except Exception as e:
            print("FID 计算失败：", e)
            print("请安装：pip install torchmetrics torch-fidelity")
    else:
        print("未安装 torchmetrics / torch-fidelity，跳过 FID。")

    print("\n所有结果已保存到：", cfg.vis_dir)


if __name__ == "__main__":
    main()
