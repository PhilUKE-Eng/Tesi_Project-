import os
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from dataloadermajorityvote import create_dataloaders

from tqdm import tqdm

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

# Percorso del modello salvato
model_path = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_resnet18FAU/train/modelresnet18FAU_epoch3.pth"  

# Caricamento del modello ResNet-50 con Dropout
print("Caricamento del modello ResNet-50 con Dropout...")
class ResNet50WithDropout(torch.nn.Module):
    def __init__(self, num_classes, fau_feature_length):
        super(ResNet50WithDropout, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Ignora il layer finale FC di ResNet50
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Total features = features di ResNet + features FAU
        total_features = in_features + fau_feature_length
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features),
            nn.ReLU(),
            nn.Linear(total_features, num_classes)
        )

    def forward(self, x, fau):
        # Estrae le features dall'immagine usando ResNet50
        x = self.resnet(x)
        # Concatena le features dell'immagine con le features FAU
        x = torch.cat((x, fau), dim=1)
        x = self.dropout(x)
        # Passa le features concatenate attraverso il classificatore
        return self.classifier(x)

# Inizializza il modello
num_classes = 6  
model = ResNet50WithDropout(num_classes, 20).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Modello caricato con successo.")

# Impostazioni per il DataLoader
dataset_image = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetcrop"
dataset_FAU = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetFAU"
train_loader, validation_loader, test_loader, num_classes = create_dataloaders(dataset_image, dataset_FAU, batch_size=32)

# Lista delle emozioni
emotions = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']

# Funzione per il majority voting
def majority_vote(predictions):
    from collections import Counter
    # print(predictions)
    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common(1)
    # print(most_common[0][0])
    return most_common[0][0], {emotions[i]: count for i, count in vote_counts.items()}


reference_labels = []
reference_video_ids = []

prediction_labels = []
prediction_video_ids = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        images = batch[0].to(device)
        fau = batch[1].to(device)
        ref_labels = batch[2]
        video_ids = batch[3]

        output = model(images, fau)
        output = torch.argmax(output, dim=-1)

        reference_labels.extend(ref_labels)
        prediction_labels.extend(output)
        reference_video_ids.extend(video_ids)
        prediction_video_ids.extend(video_ids)


# aggregate by video id
aggregated_predictions = {}
aggregated_references = {}

for i in range(len(reference_labels)):
    if reference_video_ids[i] not in aggregated_references: aggregated_references[reference_video_ids[i]] = [reference_labels[i].item()]
    else: aggregated_references[reference_video_ids[i]].append(reference_labels[i].item())

    if prediction_video_ids[i] not in aggregated_predictions: aggregated_predictions[prediction_video_ids[i]] = [prediction_labels[i].item()]
    else: aggregated_predictions[prediction_video_ids[i]].append(prediction_labels[i].item())


y_true = []
y_pred = []

for vid_id in list(set(reference_video_ids)):
    y_true.append(majority_vote(aggregated_references[vid_id])[0])
    y_pred.append(majority_vote(aggregated_predictions[vid_id])[0])
                  
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_true, y_pred))




# Calcola la matrice di confusione e il classification report
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=emotions)

# Salva la matrice di confusione come immagine
conf_matrix_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_resnet18FAU/test/confusion_matrix_majority_RSN18Fau.png'
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(conf_matrix_path)
plt.close()
print(f"Matrice di confusione salvata in {conf_matrix_path}")

# Salva il classification report in un file di testo
report_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_resnet18FAU/test/classification_report_majority_RSN18Fau.txt'
with open(report_path, 'w') as file:
    file.write(report)
print(f"Report di classificazione salvato in {report_path}")
