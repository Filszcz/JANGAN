import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
import os
from PIL import Image
import numpy as np
import random
# import albumentations as A
from lightning.pytorch.loggers import WandbLogger

wandb.finish()

torch.set_float32_matmul_precision('medium')

EPOCHS = 200

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, augment=None):
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.augment:
            image = self.augment(image)
        
        return image

class ProgressiveAugmentation:
    def __init__(self, max_epochs=100):
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
        # Spatial transforms (geometry and structure)
        self.spatial_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(
                degrees=10,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            T.RandomPerspective(
                distortion_scale=0.05,
                p=0.3
            )
        ])

        # Intensity transforms (color and texture)
        self.intensity_transforms = T.Compose([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            T.RandomPhotometricDistort(
                p=0.3,
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=(-0.1, 0.1),
                brightness=(0.8, 1.2)
            ),
            T.RandomEqualize(p=0.2),
            T.RandomPosterize(bits=7, p=0.2)
        ])

        # Noise and blur transforms
        self.noise_transforms = T.Compose([
            T.GaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0)
            ),
            T.RandomAdjustSharpness(
                sharpness_factor=1.5,
                p=0.3
            ),
            T.RandomAutocontrast(p=0.3)
        ])

        # Occlusion and final transforms
        self.augment_transforms = T.Compose([
            T.RandomErasing(
                p=0.1,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.0),
                value="random"
            ),
            T.RandomInvert(p=0.1),
            # MixUp and CutMix could be added here if needed
        ])

    def get_transform_probability(self):
        # Gradually increase augmentation strength
        return min(self.current_epoch / (self.max_epochs * 0.3), 1.0)

    def __call__(self, img):
        p = self.get_transform_probability()
        
        if random.random() < p:
            # Apply transforms sequentially with probability
            if random.random() < 0.8:  # High probability for spatial transforms
                img = self.spatial_transforms(img)
            
            if random.random() < 0.7:  # Medium-high probability for intensity
                img = self.intensity_transforms(img)
            
            if random.random() < 0.5:  # Medium probability for noise
                img = self.noise_transforms(img)
            
            if random.random() < 0.3:  # Lower probability for strong augmentations
                img = self.augment_transforms(img)
            
        return img

    def set_epoch(self, epoch):
        self.current_epoch = epoch

class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "archive",
        img_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 8,
        max_epochs: int = 100
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Base transforms (normalization, etc.)
        self.transform = T.Compose([
            T.Resize(
                size=(img_size, img_size),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            ),
            T.ToImage(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Progressive augmentation
        self.progressive_aug = ProgressiveAugmentation(max_epochs=max_epochs)

    def setup(self, stage=None):
        self.dataset = CustomImageDataset(
            img_dir=self.data_dir,
            transform=self.transform,
            augment=self.progressive_aug
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
    
    def on_train_epoch_start(self):
        # Update the current epoch in progressive augmentation
        self.progressive_aug.set_epoch(self.current_epoch)



class AdaIN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x, style):
        style = style.view(-1, 2, self.channels, 1, 1)
        gamma, beta = style[:, 0], style[:, 1]
        
        x = F.instance_norm(x)
        return gamma * x + beta

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(style_dim, style_dim),
                nn.LeakyReLU(0.2),
            ])
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        w = self.net(z)
        return w

class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(out_channels)
        self.style_proj = nn.Linear(style_dim, out_channels * 2)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        x = self.conv(x)
        style = self.style_proj(w)
        x = self.adain(x, style)
        return self.activation(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, channels=32):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim)
        
        # Initial constant input
        self.constant = nn.Parameter(torch.randn(1, channels * 8, 4, 4))
        
        # Style blocks

        self.style_blocks = nn.ModuleList([
            StyleBlock(channels * 8, channels * 8, style_dim),
            StyleBlock(channels * 8, channels * 4, style_dim),
            StyleBlock(channels * 4, channels * 4, style_dim),
            StyleBlock(channels * 4, channels * 2, style_dim),
            # StyleBlock(channels * 2, channels * 2, style_dim),
            StyleBlock(channels * 2, channels, style_dim),
        ])

        
        self.to_rgb = nn.Conv2d(channels, 3, 1)
        
    def forward(self, z):
        batch_size = z.size(0)
        w = self.mapping(z)
        
        x = self.constant.expand(batch_size, -1, -1, -1)
        
        for style_block in self.style_blocks[:-1]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = style_block(x, w)
            
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.style_blocks[-1](x, w)
       
        return torch.tanh(self.to_rgb(x))   

class Discriminator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels)
            )
        
        self.layers = nn.Sequential(
            conv_block(3, channels),
            conv_block(channels, channels * 2),
            conv_block(channels * 2, channels * 4),
            conv_block(channels * 4, channels * 8),
            conv_block(channels * 8, channels * 8),
            nn.Flatten(),
            nn.Linear(channels * 128 , 1)
        )

        
    def forward(self, x):

        return self.layers(x)

class StyleGAN(L.LightningModule):
    def __init__(
        self,
        latent_dim=512,
        style_dim=512,
        channels=128,
        lr=0.00001,
        b1=0.0,  # Changed beta1 to 0 as recommended for WGAN
        b2=0.999,
        n_critic=5,
        lambda_gp=10.0,  # Gradient penalty coefficient
        lambda_drift=0.001  # Small drift penalty to prevent drift
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Networks
        self.generator = Generator(latent_dim, style_dim, channels)
        self.discriminator = Discriminator(channels)
        
        # Loss weights
        self.lambda_gp = lambda_gp
        self.lambda_drift = lambda_drift
        
        # Logging
        self.automatic_optimization = False
        wandb.init(project="JANGAN2")
    
    def forward(self, z):
        return self.generator(z)
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty: (L2_norm(grad) - 1)**2"""
        batch_size = real_samples.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).type_as(real_samples)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get critic output for interpolated images
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).type_as(real_samples),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
        
    def training_step(self, batch, batch_idx):
        real_imgs = batch
        opt_g, opt_d = self.optimizers()
        
        # Train Critic (Discriminator)
        opt_d.zero_grad()
        
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        
        # Compute Wasserstein loss
        real_pred = self.discriminator(real_imgs)
        fake_pred = self.discriminator(fake_imgs.detach())
        
        # Wasserstein loss
        w_loss = fake_pred.mean() - real_pred.mean()
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs.detach())
        
        # Drift penalty to prevent critic output from drifting too far from 0
        drift_penalty = (real_pred ** 2).mean() * self.lambda_drift
        
        # Total critic loss
        d_loss = w_loss + self.lambda_gp * gradient_penalty + drift_penalty
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train Generator
        if batch_idx % self.hparams.n_critic == 0:
            opt_g.zero_grad()
            
            fake_pred = self.discriminator(fake_imgs)
            
            # Generator tries to minimize Wasserstein distance
            g_loss = -fake_pred.mean()
            
            self.manual_backward(g_loss)
            opt_g.step()
            
            # Logging
            self.log_dict({
                "d_loss": d_loss,
                "g_loss": g_loss,
                "gp": gradient_penalty,
                "drift": drift_penalty,
                "wasserstein_distance": -w_loss
            })
            
            if batch_idx % 100 == 0:
                wandb.log({
                    "generated_images": [wandb.Image(img) for img in fake_imgs[:4]],
                    "real_images": [wandb.Image(img) for img in real_imgs[:4]]
                })
    
    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        d_opt = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        
        schedulers = [
            CosineAnnealingLR(g_opt, T_max=200),
            CosineAnnealingLR(d_opt, T_max=200)
        ]
        
        return [g_opt, d_opt], schedulers

# Training setup

wandb_logger = WandbLogger(project="JANGAN2")

# Test the shapes
# latent_dim = 512
# batch_size = 64

# # Test Generator
# z = torch.randn(batch_size, latent_dim)
# generator = Generator()
# fake_images = generator(z)
# print(f"Generated images shape: {fake_images.shape}")  # Should be [64, 3, 128, 128]

# # Test Discriminator
# discriminator = Discriminator()
# disc_output = discriminator(fake_images)
# print(f"Discriminator output shape: {disc_output.shape}")  # Should be [64, 1]

model = StyleGAN()
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=EPOCHS,
    logger=wandb_logger,
    precision="bf16-mixed"
    # callbacks=[
    #     L.callbacks.ModelCheckpoint(
    #         monitor="g_loss",
    #         save_top_k=3,
    #         mode="min"
    #     )
    # ]
)

dm = CustomDataModule()

trainer.fit(model, dm)

wandb.finish()

