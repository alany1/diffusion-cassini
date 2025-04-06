import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import torch

class SyntheticMNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Recursively traverse the directory tree and collect image paths with their L value.
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".png"):
                    path = os.path.join(dirpath, fname)
                    # Expected filename format: "digit_{digit}_L_{L}_bg_{bg_color:.2f}.png"
                    parts = fname.split("_")
                    try:
                        # parts[3] should correspond to the L value
                        L = int(parts[3])
                    except Exception as e:
                        print(f"Error parsing L from filename: {fname}, error: {e}")
                        continue
                    self.samples.append((path, L))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, L = self.samples[idx]
        # Open the image as grayscale.
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, L


def create_dataloaders(batch_size, dataset_path, image_size=28, num_workers=4):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # scales image to [0,1]
            transforms.Lambda(lambda x: torch.log1p(x)),  # apply log(1+x) transform
            transforms.Normalize([0.3863], [0.1982]),  # suggested normalization after log1p
        ]
    )

    # Set the root folder based on your dataset structure:
    full_dataset = SyntheticMNISTDataset(root=dataset_path, transform=preprocess)

    # Split the dataset into train (80%) and validation (20%)
    total_samples = len(full_dataset)
    train_samples = int(0.8 * total_samples)
    val_samples = total_samples - train_samples
    train_dataset, val_dataset = random_split(full_dataset, [train_samples, val_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    dataset = f"{os.environ['DATASETS']}/mnist/v0"
    train_loader, val_loader = create_dataloaders(batch_size=32, dataset_path=dataset)
    for images, L_values in train_loader:
        print("Batch image shape:", images.shape)
        print("L values:", L_values)
        break

