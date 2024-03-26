from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Subset

class CelebADataLoader(BaseDataLoader):
    """
    CelebA data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        full_subset= datasets.ImageFolder(self.data_dir, transform=trsfm)
        indicies = list(range(1000))
        self.dataset = Subset(full_subset, indicies[:1000])
        #self.dataset = datasets.CelebA(self.data_dir, split='train' if training else 'test', download=False, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)