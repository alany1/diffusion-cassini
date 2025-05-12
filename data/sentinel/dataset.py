import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch

class SyntheticSentinelDataset(Dataset):
    def __init__(self, root, clean_L: int, transform=None):
        """
        
        :param root: 
        :param clean_L: value to use for L if the image is clean (technically infty) 
        :param transform: 
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.clean_L = clean_L

        # Recursively traverse the directory tree and collect image paths with their L value.
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".png"):
                    path = os.path.join(dirpath, fname)

                    # Expected filename format: "terrain_{terrain}_L_{L}_id{id}.png"
                    parts = fname.split("_")
                    try:
                        # parts[3] should correspond to the L value
                        if parts[3] == "None":
                            L = clean_L
                        else:
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
        return image, L, path


def create_dataloaders(*, batch_size, test_batch_size, dataset_path, clean_L, image_size=112, num_workers=4):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # scales image to [0,1]
            transforms.Lambda(lambda x: torch.log1p(x)),  # apply log(1+x) transform
            transforms.Normalize([0.3863], [0.1982]),  # suggested normalization after log1p
        ]
    )

    # Set the root folder based on your dataset structure:
    full_dataset = SyntheticSentinelDataset(root=dataset_path, clean_L=clean_L, transform=preprocess)

    # Split the dataset into train (80%) and validation (20%)
    total_samples = len(full_dataset)
    train_samples = int(0.8 * total_samples)
    val_samples = total_samples - train_samples
    train_dataset, val_dataset = random_split(full_dataset, [train_samples, val_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def create_eval_dataloader(*, test_batch_size, dataset_path, clean_L, image_size=28, num_workers=4):
    pass

# Example usage:
if __name__ == "__main__":
    dataset = f"{os.environ['DATASETS']}/sentinel/noised/v1/agri"
    train_loader, val_loader = create_dataloaders(batch_size=32, test_batch_size=32, clean_L=100, dataset_path=dataset)
    x = next(iter(train_loader))
    
    x[0]
    x[0].shape
    x[1]
    
    print("Done")

    # debug
    # for images, L_values in train_loader:
    #     print("Batch image shape:", images.shape)
    #     print("L values:", L_values)
    #     break

