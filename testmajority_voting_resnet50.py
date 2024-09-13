import os
import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

# Percorso al dataset e al modello salvato
base_path = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetcrop/test/"
model_path = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res50crop/train2/modelresnet50_epoch4.pth"  

# Caricamento del modello ResNet-50 con Dropout
print("Caricamento del modello ResNet-50 con Dropout...")
class ResNet50WithDropout(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithDropout, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

num_classes = 6  
model = ResNet50WithDropout(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Modello caricato con successo.")

# Trasformazioni per le immagini
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lista delle emozioni
emotions = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']

# Funzione per il majority voting
def majority_vote(predictions):
    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common(1)
    return most_common[0][0], {emotions[i]: count for i, count in vote_counts.items()}

# Funzione per prevedere la classe di un'immagine
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Inizializza una lista per salvare i risultati
results = []

# Variabili per calcolare la percentuale di predizioni corrette
correct_predictions = 0
total_videos = 0
all_labels = []
all_preds = []

# Elaborazione delle immagini per ogni emozione e attore
for emotion in emotions:
    emotion_path = os.path.join(base_path, emotion)
    if os.path.isdir(emotion_path):
        for actor in os.listdir(emotion_path):
            actor_path = os.path.join(emotion_path, actor)
            if os.path.isdir(actor_path):
                video_frames = {}
                for filename in os.listdir(actor_path):
                    if filename.endswith('.jpg'):
                        frame_id = filename.split('-')[-1][:4]
                        video_id = filename[:-9]
                        
                        if video_id not in video_frames:
                            video_frames[video_id] = []
                        
                        image_path = os.path.join(actor_path, filename)
                        prediction = predict(image_path)
                        video_frames[video_id].append(prediction)
                
                for video_id, predictions in video_frames.items():
                    final_prediction, vote_counts = majority_vote(predictions)
                    print(f"Video ID: {video_id}, Actor: {actor}, Emotion: {emotion}")
                    print(f"Voti per frame: {vote_counts}")
                    print(f"Predizione finale (majority vote): {emotions[final_prediction]}")
                    
                    is_correct = (emotions[final_prediction] == emotion)
                    correct_predictions += int(is_correct)
                    total_videos += 1

                    all_labels.append(emotions.index(emotion))
                    all_preds.append(final_prediction)
                    
                    results.append({
                        'video_id': video_id,
                        'actor': actor,
                        'emotion': emotion,
                        'final_prediction': emotions[final_prediction],
                        'vote_counts': vote_counts,
                        'correct': is_correct
                    })

# Calcola la percentuale di predizioni corrette
accuracy = (correct_predictions / total_videos) * 100
print(f"Percentuale di predizioni corrette: {accuracy:.2f}%")

# Crea un DataFrame e salva i risultati in un file CSV
output_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res50crop/test2/results_majority_voting_resnet50_epoch8.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"Risultati salvati in {output_path}")

# Genera la confusion matrix e il classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=emotions)

# Salva la confusion matrix in un file PNG
conf_matrix_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res50crop/test2/confusion_matrix_epoch8.png'
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(conf_matrix_path)
plt.close()
print(f"Matrice di confusione salvata in {conf_matrix_path}")

# Salva il classification report in un file di testo
class_report_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res50crop/test2/classification_report_epoch8.txt'
with open(class_report_path, 'w') as f:
    f.write(class_report)
print(f"Report di classificazione salvato in {class_report_path}")
