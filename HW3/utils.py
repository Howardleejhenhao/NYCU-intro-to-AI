import PIL.Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name

def load_train_dataset(path: str = 'data/train/') -> Tuple[List[str], List[int]]:
    images = []
    labels = []
    label_mapping = {
        "elephant": 0,
        "jaguar": 1,
        "lion": 2,
        "parrot": 3,
        "penguin": 4
    }
    for animal, label in label_mapping.items():
        fol_path = os.path.join(path, animal)
        for filename in os.listdir(fol_path):
            each_path = os.path.join(fol_path, filename)
            images.append(each_path)
            labels.append(label)
    return images, labels

def load_test_dataset(path: str = 'data/test/') -> List[str]:
    images = []
    for filename in os.listdir(path):
        each_path = os.path.join(path, filename)
        images.append(each_path)
    return images


def plot(train_losses: List, val_losses: List):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    print("Save the plot to 'loss.png'")
    return