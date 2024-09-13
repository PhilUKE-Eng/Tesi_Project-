# dataloader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.transform = transform
        
        for label, emotion in enumerate(self.class_names):
            emotion_dir = os.path.join(root_dir, emotion)
            for actor in os.listdir(emotion_dir):
                actor_dir = os.path.join(emotion_dir, actor)
                images = [os.path.join(actor_dir, img) for img in os.listdir(actor_dir) if img.endswith(".jpg")]
                self.image_paths.extend(images)
                self.labels.extend([label] * len(images))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def create_dataloaders(root_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    
    phases = ['train', 'test', 'validation']
    dataloaders = {}
    num_classes = None

    for phase in phases:
        dataset = EmotionDataset(os.path.join(root_dir, phase), transform=transform)
        dataloaders[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        if num_classes is None:
            num_classes = len(dataset.class_names)
    
    return dataloaders['train'], dataloaders['validation'], dataloaders['test'], num_classes






