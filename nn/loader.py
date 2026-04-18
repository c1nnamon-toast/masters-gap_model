import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SkyImageDataset(Dataset):
    """Dataset for sky images with irradiance labels."""
    
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if (torch.is_tensor(idx)):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        picture_name = row['PictureName']
        
        img_path = os.path.join(self.image_dir, picture_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        irradiance = row['IrradianceToPredict']
        label = torch.tensor(irradiance, dtype=torch.float32).view(1)
        
        return image, label


def get_transforms(normalize_mean=None, normalize_std=None):
    """Get image transformation pipeline"""

    transform_list = []
    transform_list.append(transforms.ToTensor())

    # Normalization
    if normalize_mean and normalize_std:
        transform_list.append(
            transforms.Normalize(mean=normalize_mean, std=normalize_std))
    
    return transforms.Compose(transform_list)


def load_csv(csv_path):
    """Load dataframe from CSV and add IrradianceToPredict column"""

    df = pd.read_csv(csv_path)
    
    # Add IrradianceToPredict column for consistency
    df['IrradianceToPredict'] = df['Irradiance']
    
    return df


def get_dataloaders(train_csv, val_csv, test_csv, 
                    train_image_dir, val_image_dir, test_image_dir,
                    batch_size, num_workers, pin_memory,
                    normalize_mean=None, normalize_std=None):
    """Create train/val/test data loaders from pre-split dataset."""

    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)
    test_df = load_csv(test_csv)
    
    train_transform = get_transforms(normalize_mean=normalize_mean,
                                     normalize_std=normalize_std)
    eval_transform = get_transforms(normalize_mean=normalize_mean,
                                    normalize_std=normalize_std)
    
    train_dataset = SkyImageDataset(train_df, train_image_dir, train_transform)
    val_dataset = SkyImageDataset(val_df, val_image_dir, eval_transform)
    test_dataset = SkyImageDataset(test_df, test_image_dir, eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    # Combine all splits for EDA
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    return train_loader, val_loader, test_loader, full_df
