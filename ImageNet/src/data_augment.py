import numpy as np
import torch
from torchvision import transforms, datasets
from PIL import Image
from dataset import ImageNetKaggle
import os
import pickle


class PCAColorAugmentation:
    def __init__(self, eigenvectors, eigenvalues, alpha_std=0.2):
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.alpha_std = alpha_std

    def __call__(self, img):
        """
        Perform PCA-based color augmentation on an image.

        Args:
            img: PIL Image in RGB format.

        Returns:
            Augmented PIL Image.
        """
        img_array = np.array(img).astype(np.float32) / 255.0  # Scale to [0, 1]
        img_flat = img_array.reshape(-1, 3)  # Flatten to (N, 3)

        # Generate random noise
        alphas = np.random.normal(0, self.alpha_std, size=3)
        delta = np.dot(self.eigenvectors, np.sqrt(self.eigenvalues) * alphas)

        # Apply the shift
        img_aug = img_flat + delta
        img_aug = np.clip(img_aug, 0, 1).reshape(img_array.shape)

        # Convert back to [0, 255] and uint8
        img_aug = (img_aug * 255).astype(np.uint8)

        return Image.fromarray(img_aug)

class ImageNetDataAugmentation:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    DO_PCA = False

    def __init__(self, data_path="../data"):
        """
        Computes the eigenvalues and eigenvectors of the ImageNet dataset for PCA-based color augmentation.
        Provides the train and test transforms for the ImageNet dataset.

        Args:
            data_path: Path to the ImageNet dataset.
        
        """
        self.data_path = data_path
        if self.DO_PCA:
            pca_file = os.path.join(data_path, "pca_data.pkl")
            self.eigenvectors, self.eigenvalues = None, None
            if os.path.exists(pca_file):
                with open(pca_file, "rb") as f:
                    data = pickle.load(f)
                self.eigenvectors, self.eigenvalues = data["eigenvectors"], data["eigenvalues"]
                print("Loaded PCA data from file.")
            else:
                print("Computing eigenvalues and eigenvectors of the RGB covariance matrix of the dataset...")
                self.eigenvectors, self.eigenvalues = self.compute_global_pca()
                with open(pca_file, "wb") as f:
                    pickle.dump({"eigenvectors": self.eigenvectors, "eigenvalues": self.eigenvalues}, f)
                print("Saved PCA data to file.")
            
            self.pca_augment = PCAColorAugmentation(self.eigenvectors, self.eigenvalues)
                

    def compute_global_pca(self):
        transform = transforms.Compose([
            transforms.ToTensor()  # Just load as tensor for now
        ])
        train_dataset = ImageNetKaggle(self.data_path, "train", transform)

        all_pixels = []

        for img, _ in train_dataset:
            img_array = np.array(transforms.ToPILImage()(img)) / 255.0  # Convert to [0, 1]
            img_flat = img_array.reshape(-1, 3)  # Flatten to (N, 3)
            all_pixels.append(img_flat)

        all_pixels = np.vstack(all_pixels)  # Stack all pixels across the dataset
        mean = np.mean(all_pixels, axis=0)
        all_pixels -= mean  # Center the data

        cov = np.cov(all_pixels, rowvar=False)  # Compute covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order
        sort_perm = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort_perm]
        eigenvectors = eigenvectors[:, sort_perm]

        return eigenvectors, eigenvalues

    def get_train_transforms(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),  # Random crop of size 224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15,
                scale=(0.85, 1.3)
            ),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Might be causing slow down?
            transforms.Lambda(lambda img: self.pca_augment(img)) if self.DO_PCA else transforms.Lambda(lambda img: img),  # PCA Color Augmentation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=self.MEAN, std=self.STD)  # Normalize
        ])


    def get_test_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(self.MEAN, self.STD)(crop) for crop in crops]))
        ])


# Usage
data_augmentation = ImageNetDataAugmentation()
train_transforms = data_augmentation.get_train_transforms()
test_transforms = data_augmentation.get_test_transforms()
