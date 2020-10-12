import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
import os
from PIL import Image
import os.path as osp
from utils.transform import *


# EndoScene Dataset
class EndoScene(Dataset):
    def __init__(self, root, data_dir, mode='train', transform=None):
        super(EndoScene, self).__init__()
        data_path1 = osp.join(root, data_dir) + '/CVC-300'
        data_path2 = osp.join(root, data_dir) + '/CVC-612'
        self.imglist = []
        self.gtlist = []

        datalist1 = os.listdir(osp.join(data_path1, 'image'))
        for data1 in datalist1:
            self.imglist.append(osp.join(data_path1 + '/image', data1))
            self.gtlist.append(osp.join(data_path1 + '/gtpolyp', data1))

        datalist2 = os.listdir(osp.join(data_path2, 'image'))
        for data2 in datalist2:
            self.imglist.append(osp.join(data_path2 + '/image', data2))
            self.gtlist.append(osp.join(data_path2 + '/gtpolyp', data2))

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((288, 384)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #Translation(10),
                   RandomCrop((256, 256)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((288, 384)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.imglist)
