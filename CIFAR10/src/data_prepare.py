import os
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

# Define the directory where augmented data will be saved
output_dir = "../offline_augmented_data"
os.makedirs(output_dir, exist_ok=True)

# Original dataset
transform = transforms.Compose([
    transforms.ToTensor()  # Just load as tensor for now
])
train_dataset = datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)

# Compute global PCA

def compute_global_pca(dataset):
    """
    Compute the global PCA on the RGB pixel values of the dataset.

    Args:
        dataset: Torch dataset with images.

    Returns:
        eigenvectors, eigenvalues: Global PCA components for the dataset.
    """
    all_pixels = []

    for img, _ in dataset:
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

eigenvectors, eigenvalues = compute_global_pca(train_dataset)

# PCA-based color augmentation function
def pca_color_augmentation(image, eigenvectors, eigenvalues, alpha_std=0.2):
    """
    Perform PCA-based color augmentation using precomputed global PCA components.

    Args:
        image: PIL Image in RGB format.
        eigenvectors: Precomputed eigenvectors from global PCA.
        eigenvalues: Precomputed eigenvalues from global PCA.
        alpha_std: Standard deviation for random noise applied to the principal components.

    Returns:
        Augmented PIL Image.
    """
    img_array = np.array(image).astype(np.float32)  # Convert to numpy array
    img_array /= 255.0  # Scale to [0, 1]

    img_flat = img_array.reshape(-1, 3)  # Flatten image to (N, 3)

    # Generate random noise
    alphas = np.random.normal(0, alpha_std, size=3)
    delta = np.dot(eigenvectors, np.sqrt(eigenvalues) * alphas)


    # Apply the shift
    img_aug = img_flat + delta
    img_aug = np.clip(img_aug, 0, 1).reshape(img_array.shape)

    # Convert back to [0, 255] and uint8
    img_aug = (img_aug * 255).astype(np.uint8)

    return Image.fromarray(img_aug)




# Define augmentations for additional copies
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=15,  # Random rotation within +/- 15 degrees
        translate=(0.15, 0.15),  # Random translation
        scale=(0.7, 1.3)  # Zoom-in only (scale >= 1.0)
    )
])

# Make directories for each class
for class_id in range(10):
    class_dir = os.path.join(output_dir, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

# Loop through the original dataset
count = 0

for idx in range(len(train_dataset)):
    img, label = train_dataset[idx]

    # Undo normalization for PIL processing

    pil_img = transforms.ToPILImage()(img)

    # Apply PCA-based augmentation using global PCA components
    pca_augmented_img = pil_img# pca_color_augmentation(pil_img, eigenvectors, eigenvalues)


    # Save the PCA-augmented original image
    out_path = os.path.join(output_dir, str(label), f"img_{count:06d}.png")
    pca_augmented_img.save(out_path)
    count += 1

    # Create 3 augmented versions (including PCA-based color augmentation)
    for _ in range(5):
        # Apply augmentations
        aug_img = augmentation_transforms(pil_img)

        # Apply PCA color augmentation
        aug_img = pca_color_augmentation(aug_img, eigenvectors, eigenvalues)

        # Save the augmented image
        out_path = os.path.join(output_dir, str(label), f"img_{count:06d}.png")
        aug_img.save(out_path)
        count += 1

print("Offline augmentation complete!")
print(f"Total images: {count}")
