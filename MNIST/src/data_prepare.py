import os
import torch
from torchvision import datasets, transforms
from PIL import Image

# Define the directory where augmented data will be saved
output_dir = "offline_augmented_data"
os.makedirs(output_dir, exist_ok=True)

# Original dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Just load as tensor for now
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)

# Define your augmentations that you want to apply offline
# We'll use transforms that operate on PIL Images, so we convert tensors back to PIL.
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        scale=(0.8, 1.2)  # randomly scale between 80% and 120%
    )
])

# We will produce 4 images per original image: 1 original normalized and 3 augmented variants.
# If you'd like the original also to be augmented or processed differently, you can do that too.

for class_id in range(10):
    class_dir = os.path.join(output_dir, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

# Loop through the original dataset
count = 0
for idx in range(len(train_dataset)):
    img, label = train_dataset[idx]
    # img is a normalized tensor, but we want to apply PIL-based transforms
    # Undo normalization to get back original range if needed:
    # Current normalization: (x * 0.5) + 0.5 to undo
    # We'll just convert directly to a PIL image from the tensor in [-1,1].
    # PIL expects [0,1], so we do (img*0.5)+0.5 to restore original range:
    img_unorm = (img * 0.5) + 0.5
    pil_img = transforms.ToPILImage()(img_unorm)
    pil_img = pil_img.convert("L")  # Convert to grayscale explicitly

    # Save the original image as one variant
    out_path = os.path.join(output_dir, str(label), f"img_{count:06d}.png")
    pil_img = transforms.ToPILImage(mode='L')(img_unorm)
    pil_img.save(out_path)
    count += 1

    # Create 3 more augmented versions
    for _ in range(3):
        aug_img = augmentation_transforms(pil_img)
        aug_img = aug_img.convert("L")
        out_path = os.path.join(output_dir, str(label), f"img_{count:06d}.png")
        aug_img.save(out_path)
        count += 1

print("Offline augmentation complete!")
print(f"Total images: {count}")
