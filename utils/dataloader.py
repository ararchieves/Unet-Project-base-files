import os 
import sys
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.io import read_image, ImageReadMode


class DHADataset(Dataset):
    def __init__(self, base_dir='data', split='train', transform=None):
        super().__init__()
        self.base_dir = f"{base_dir}/{split}"
        self.images = os.listdir(f"{self.base_dir}/images")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]

        image = read_image(f"{self.base_dir}/images/{filename}", mode=ImageReadMode.RGB) / 255
        house = read_image(f"{self.base_dir}/houses/{filename}", mode=ImageReadMode.GRAY) / 255
        block = read_image(f"{self.base_dir}/blocks/{filename}", mode=ImageReadMode.GRAY) / 255

        if self.transform:
            image = self.transform(image)
            house = self.transform(house)
            block = self.transform(block)

        return image, house, block
    

if __name__ == "__main__":
    try:
        trainset = DHADataset('dha_dataset')
        testset = DHADataset('dha_dataset', split='test')
    except FileNotFoundError:
        print("No Dataset Found.")
        sys.exit(0)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    
    print(f"Length of trainset: {len(trainset)} - Length of testset: {len(testset)}")

    images, houses, blocks = next(iter(trainloader))
    
    print(f"images: {images.shape} - houses: {houses.shape} - blocks: {blocks.shape}")