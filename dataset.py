import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T


# Normalization parameters for pre-trained PyTorch models (from ImageNet data set)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root: str = "crypko_data/faces", hr_size: int = 96):
        """
        Args:
            root (str, optional): [description].
                Defaults to "crypko_data/faces".
            hr_size (int, optional): [description].
                    Defaults to 96.
        """
        hr_height, hr_width = hr_size
        self.files = glob.glob(root + "/*.*")

        self.lr_transform = T.Compose([
            T.Resize((hr_height // 4, hr_width // 4), interpolation = Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        self.hr_transform = T.Compose([
            T.Resize((hr_height, hr_width), interpolation = Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __getitem__(self, index: int):
        """Get the data in the dataset(low / high resolution)
        Args:
            index (int): Index of the file to get.
        Returns:
            [Dict]:
                'lr': [Tensor]: Low resolution image
                'hr': [Tensor]: High resolution image
        """
        img = Image.open(self.files[index % len(self.files)])
        lr_img = self.lr_transform(img)
        hr_img = self.hr_transform(img)

        return {"lr": lr_img, "hr": hr_img}

    def __len__(self):
        return len(self.files)


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.N = len(data_source)


    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx


def get_dataloader(root_dir: str = "crypko_data/faces", batch_size: int = 16):
    """
    Args:
        root_dir (str, optional): [description]. Defaults to "crypko_data/faces".
        batch_size (int, optional): [description]. Defaults to 16.

    Returns:
        [Dataloader]
    """
    dataset = ImageDataset(root_dir)
    sampler = InfiniteSampler(dataset)
    return DataLoader(dataset, batch_size = batch_size, sampler = sampler)