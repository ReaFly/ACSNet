import os
import os.path as osp
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms


# KavSir-SEG Dataset
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(kvasir_SEG, self).__init__()
        data_path = osp.join(root, data2_dir)
        self.imglist = []
        self.gtlist = []

        datalist = os.listdir(osp.join(data_path, 'images'))
        for data in datalist:
            self.imglist.append(osp.join(data_path+'/images', data))
            self.gtlist.append(osp.join(data_path+'/masks', data))

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((320, 320)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #Translation(10),
                   RandomCrop((224, 224)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
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
