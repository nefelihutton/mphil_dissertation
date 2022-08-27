"""

U-Net with different backbones taken from segmentation models pytorch module
Specify the encoder in the train_net() or as an arg.
ImageNet encoder weights are used - 
in our experiments, models with the imagenet pretrained weights had higher performance

"""


import argparse
import logging
import sys
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
from utils.dataloader import BinaryDataset
from torch.utils.data import DataLoader, random_split

mask_dir = "data/train/masks"
img_dir = "data/train/images"

test_mask_dir = "data/test/masks"
test_img_dir = "data/test/images"

img_scale = float(1.0)
val_percent = float(0.5)
batch_size = int(16)
epochs = int(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_net(net,
              device,
              epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 1e-5,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    dataset = BinaryDataset(img_dir, mask_dir, img_scale)
    test_set = BinaryDataset(img_dir, mask_dir, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # 3. Create data loaders
    n_cpu = os.cpu_count()

    loader_args = dict(batch_size=batch_size, num_workers=n_cpu, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)

    # 4. Load model
    encoder = "resnet34"
    encoder_wts = "imagenet"
    activation = "sigmoid"

    model = smp.Unet(encoder_name=encoder,activation=activation,encoder_weights=encoder_wts)
    preprocess_func = smp.encoders.get_preprocessing_fn(encoder,encoder_wts)

    # 4. Set up the optimizer, the loss and the learning rate scheduler
    trainmodel = True
    epochs = 20
    diceloss = smp.utils.losses.DiceLoss()
    metrics = [ smp.utils.metrics.IoU(threshold=0.5) ]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001)])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 5. Begin training
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        # training phase
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # forward
                outputs = model(images)
                loss = diceloss(outputs, true_masks)
                # cal loss and backward
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
  
        # val phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            with tqdm(total=n_val, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                images = batch['image']
                true_masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                outputs = model(images)
                loss = diceloss(outputs, true_masks)
                valid_loss += loss.item()
            valid_loss /= len(val_loader)
    print(f'EPOCH: {epoch + 1} - train loss: {train_loss} -  valid_loss: {valid_loss}')
    torch.save(model.state_dict(), f'model_{encoder}.pth')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet-ResNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--enconder', '-enc', type=str, default='resnet32', help='encoder from SMP')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--activation', '-a', type=str, default='sigmoid')
    parser.add_argument('--encoder_wts', '-w', type=str, default='imagenet')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = smp.Unet(encoder_name=args.encoder,activation=args.activation,encoder_weights=args.encoder_wts)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_net(net=model,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
