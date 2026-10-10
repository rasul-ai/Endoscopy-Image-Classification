import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, class_names=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        if class_names is None:
            # sorted() guarantees deterministic, reproducible class ordering
            # across any machine or filesystem state
            class_names = sorted(
                [d for d in os.listdir(data_path)
                 if os.path.isdir(os.path.join(data_path, d))]
            )

        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(data_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, image_file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
