import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset  #  utils. 제거
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/'
        model,
        device,
        epochs: int = 80,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    dataset = BasicDataset(dir_img, dir_mask, img_scale))

def train_model(
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()  #  Binary Segmentation 손실 함수
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.float32)  #  float 타입으로 변환

                masks_pred = model(images)  # logits 값
                loss = criterion(masks_pred.squeeze(1), true_masks.squeeze(1))   #  BCEWithLogitsLoss 적용
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.squeeze(1), multiclass=False) #  sigmoid 적용 후 Dice Loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)  #  n_classes=1 로 고정

    train_model(
        model=model,
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
        device=device
    )
