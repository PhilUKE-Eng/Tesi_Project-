import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, root_dir, fau_root, transform=None, is_test=False):
        self.root_dir = root_dir
        self.fau_root = fau_root
        self.transform = transform
        self.is_test = is_test
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.list_fau_names = [
            "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", 
            "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"
        ]

        # Prepara gli elenchi di immagini e etichette
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
        # Caricamento immagine
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Caricamento delle feature FAU
        fau_path = image_path.replace(self.root_dir, self.fau_root).replace(".jpg", ".csv")
        fau_data = pd.read_csv(fau_path)

        # Seleziona solo le colonne FAU presenti in list_fau_names
        fau_data = fau_data[self.list_fau_names]

        # Riempie eventuali NaN con 0
        fau_data = fau_data.fillna(0)

        # Converti i dati FAU in tensore di PyTorch
        fau_features = torch.tensor(fau_data.values.astype('float32')).flatten()

        # Verifica che il numero di feature FAU sia corretto (20 in questo esempio)
        if fau_features.shape[0] != 20:
            print(f"Dimensione fau_features diversa da 20: {fau_features.shape}, file: {fau_path}")
            fau_features = fau_features[:20]  # Taglia a 20 se necessario

        # Etichetta
        label = self.labels[idx]

        # Restituisci il video ID se siamo nel test set
        if self.is_test:
            video_id = self.extract_video_id(image_path)
            return image, fau_features, torch.tensor(label), video_id
        else:
            return image, fau_features, torch.tensor(label)

    def extract_video_id(self, image_path):
        # Estrazione dell'ID del video dal nome del file o dalla struttura del percorso
        video_id = os.path.basename(image_path).split('_')[0]
        video_id = video_id.split(".")[0]
        video_id = video_id.split("-")[:-1]
        video_id = "-".join(video_id)
        return video_id


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
        dataset = EmotionDataset(dataset_path, os.path.join(fau_root, phase), transform=transform, is_test=phase=="test")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloaders[phase] = dataloader
        if num_classes is None:
            num_classes = len(dataset.class_names)
    
    return dataloaders['train'], dataloaders['validation'], dataloaders['test'], num_classes
