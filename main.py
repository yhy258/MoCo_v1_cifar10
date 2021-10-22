import tarfile
from config import Config
import os
from models import *
from dataset import CIFAR_CUSTOM_DATASET
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

import numpy as np
import math

def main():
    """
        CIFAR10 기준. dataset을 custom 했었어야해서 cifar10.tar.gz data를 기준으로 작성.
    """
    cfg = Config()

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    fname = cfg.cifar_tar_gz_name
    ap = tarfile.open(fname)

    directory = cfg.directory
    createFolder(directory)

    ap.extractall(directory)
    ap.close()

    def cosine_learning_rate(optimizer, lr, epoch, epochs=50):
        """Decay the learning rate based on schedule"""
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MoCo().to(DEVICE)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    train_transform = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
    dataset = CIFAR_CUSTOM_DATASET(cfg.directory, transforms.Compose(train_transform))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.EPOCH+1):
        print("epoch : {}/{}".format(epoch, cfg.EPOCH))
        cosine_learning_rate(optimizer, cfg.lr, epoch, cfg.EPOCH)
        losses = []
        for img1, img2 in tqdm(dataloader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            logit, label = model(img1, img2)

            loss = criterion(logit, label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss : {}".format(np.mean(losses)))

if __name__ == "__main__":
    main()