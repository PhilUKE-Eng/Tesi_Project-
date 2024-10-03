import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    def __init__(self, root_dir, fau_root, transform=None, is_test=False):
        self.root_dir = root_dir
        self.fau_root = fau_root
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.transform = transform
        self.is_test = is_test
        self.list_fau_names = [
            "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"

        ]
        
        # Preparazione degli elenchi di immagini e etichette
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
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        fau_path = image_path.replace(self.root_dir, self.fau_root).replace(".jpg", ".csv")
        fau_data = pd.read_csv(fau_path)

        # Rimuovere le colonne 'input' e 'frame' se presenti
        #fau_data = fau_data.drop(columns=['input', 'frame'], errors='ignore')
        fau_data = fau_data[self.list_fau_names]
        # Selezionare solo colonne numeriche
        fau_data = fau_data.select_dtypes(include=[np.number])
        
        # Controllo degli elementi NaN e sostituzione con 0 
        # if fau_data.isna().any().any():  # Verifica se ci sono NaN
        #     print(f"Valori NaN trovati in {fau_path}, sostituendo con 0.")
        fau_data = fau_data.fillna(0)  
        
        fau_features = torch.tensor(fau_data.values.astype('float32')).flatten()

        # Controllo della dimensione delle feature e taglio a 683 elementi se necessario
        if fau_features.shape[0] != 20:
            print(f"Dimensione fau_features diversa da 683: {fau_features.shape}, file: {fau_path}")
            fau_features = fau_features[:20]  # Tagliare a 683 elementi

        label = self.labels[idx]

        

        if self.is_test:
            # TODO: get video id
            video_id = 0
            return image, fau_features, torch.tensor(label), video_id
        else:
            return image, fau_features, torch.tensor(label)


    # @staticmethod
    # def collate_fn(batch):
    #     images, fau_features, labels = zip(*batch)
    #     images = torch.stack(images)
    #     fau_features = pad_sequence([torch.tensor(f) for f in fau_features], batch_first=True, padding_value=0)
    #     labels = torch.tensor(labels)
    #     return images, fau_features, labels

def create_dataloaders(root_dir, fau_root, batch_size=32, transform=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    phases = ['train', 'test', 'validation']
    dataloaders = {}
    num_classes = None

    for phase in phases:
        dataset_path = os.path.join(root_dir, phase)
        dataset = EmotionDataset(dataset_path, os.path.join(fau_root, phase), transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, is_test=phase=="test")
        dataloaders[phase] = dataloader
        if num_classes is None:
            num_classes = len(dataset.class_names)
    
    return dataloaders['train'], dataloaders['validation'], dataloaders['test'], num_classes

# Uso del DataLoader:
# dataset_image = "/path/to/images"
# dataset_FAU = "/path/to/fau"
# train_loader, validation_loader, test_loader, num_classes = create_dataloaders(dataset_image, dataset_FAU, batch_size=32)
