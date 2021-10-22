import os
from PIL import Image
import torch
from torch.utils.data import Dataset

"""
    
"""



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpickle_all(cifar_directory):
    data1 = unpickle(os.path.join(cifar_directory, "data_batch_1"))
    data2 = unpickle(os.path.join(cifar_directory, "data_batch_2"))
    data3 = unpickle(os.path.join(cifar_directory, "data_batch_3"))
    data4 = unpickle(os.path.join(cifar_directory, "data_batch_4"))
    data5 = unpickle(os.path.join(cifar_directory, "data_batch_5"))

    return [data1, data2, data3, data4, data5]


class CIFAR_CUSTOM_DATASET(Dataset):
    def __init__(self, directory, transform):
        super().__init__()
        self.transform = transform
        cifar_directory = os.path.join(directory, "cifar-10-batches-py")
        datas = unpickle_all(cifar_directory)
        # all = [np.zeros((50000, 32, 32, 3))]
        # for i,data in enumerate(datas):
        #     all[i*10000: 10000*(i+1)] = data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        all = []
        for data in datas:
            for img_ in data[b'data']:
                all.append(Image.fromarray(img_.reshape(3, 32, 32).transpose(1, 2, 0)))
        self.all = all

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = self.all[idx]

        if self.transform:
            img1 = self.transform(imgs)  # random crop 두번 다 다른 결과
            img2 = self.transform(imgs)

        return img1, img2

    def __len__(self):
        return len(self.all)