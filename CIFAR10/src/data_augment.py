import numpy as np
import torch
from torchvision import transforms, datasets
from PIL import Image


torch.set_float32_matmul_precision('high')


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

class CIFAR10DataAugmentation:
    MEAN = 0.475
    STD = 0.25


    def __init__(self, data_path="../data"):
        """
        Computes the eigenvalues and eigenvectors of the CIFAR-10 dataset for PCA-based color augmentation.
        Provides the train and test transforms for the CIFAR-10 dataset.

        Args:
            data_path: Path to the CIFAR-10 dataset.
        
        """
        self.data_path = data_path
        self.eigenvectors, self.eigenvalues = self.compute_global_pca()
        self.pca_augment = PCAColorAugmentation(self.eigenvectors, self.eigenvalues)

    def compute_global_pca(self):
        transform = transforms.Compose([
            transforms.ToTensor()  # Just load as tensor for now
        ])
        train_dataset = datasets.CIFAR10(root=self.data_path, train=True, transform=transform, download=True)

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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.15, 0.15),
                scale=(0.85, 1.3)
            ),
            transforms.Lambda(lambda img: self.pca_augment(img)),  # PCA Color Augmentation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(self.MEAN, self.MEAN, self.MEAN), std=(self.STD, self.STD, self.STD))  # Dont normalize here to avoid 32-bit conversion
        ])

    def get_test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(self.MEAN, self.MEAN, self.MEAN), std=(self.STD, self.STD, self.STD))
        ])

# Usage
data_augmentation = CIFAR10DataAugmentation()
train_transforms = data_augmentation.get_train_transforms()
test_transforms = data_augmentation.get_test_transforms()
