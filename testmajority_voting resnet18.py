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
model_path = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res18crop/train1/modelresnet18_epoch2.pth" 

# Caricamento del modello
print("Caricamento del modello ResNet-18...")
model = models.resnet18(weights=None)
num_classes = 6  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
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
    return most_common[0][0], vote_counts

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

# Liste per le etichette vere e predette
true_labels = []
pred_labels = []

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
                        frame_id = filename.split('-')[-1][:4]  # Prendi le ultime 4 cifre
                        video_id = filename[:-9]  # Nome del file senza le ultime 4 cifre e l'estensione
                        
                        if video_id not in video_frames:
                            video_frames[video_id] = []
                        
                        image_path = os.path.join(actor_path, filename)
                        prediction = predict(image_path)
                        video_frames[video_id].append(prediction)
                
                # Applica il majority vote per ogni video
                for video_id, predictions in video_frames.items():
                    final_prediction, vote_counts = majority_vote(predictions)
                    print(f"Video ID: {video_id}, Actor: {actor}, Emotion: {emotion}")
                    print(f"Voti per frame: {vote_counts}")
                    print(f"Predizione finale (majority vote): {emotions[final_prediction]}")
                    
                    # Verifica se la predizione finale è corretta
                    is_correct = (emotions[final_prediction] == emotion)
                    correct_predictions += int(is_correct)
                    total_videos += 1
                    
                    # Aggiungi le etichette vere e predette alle liste
                    true_labels.append(emotions.index(emotion))
                    pred_labels.append(final_prediction)
                    
                    # Prepara una stringa per i voti per frame con i nomi delle etichette
                    vote_counts_str = ", ".join([f"{emotions[label]}: {count}" for label, count in vote_counts.items()])
                    
                    results.append({
                        'video_id': video_id,
                        'actor': actor,
                        'emotion': emotion,
                        'final_prediction': emotions[final_prediction],
                        'correct': is_correct,
                        'vote_counts': vote_counts_str
                    })

# Calcola la percentuale di predizioni corrette
accuracy = (correct_predictions / total_videos) * 100
print(f"Percentuale di predizioni corrette: {accuracy:.2f}%")

# Crea un DataFrame e salva i risultati in un file CSV
output_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res18crop/test1/results_majority_voting_resnet18-4.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)

print(f"Risultati salvati in {output_path}")

# Calcola e salva la matrice di confusione
cm = confusion_matrix(true_labels, pred_labels)
cm_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res18crop/test1/confusion_matrix_resnet18-4.png'
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(cm_path)
plt.close()
print(f"Matrice di confusione salvata in {cm_path}")

# Calcola e salva il classification report
report = classification_report(true_labels, pred_labels, target_names=emotions)
report_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res18crop/test1/classification_report_resnet18-4.txt'
with open(report_path, 'w') as f:
    f.write(report)
print(f"Classification report salvato in {report_path}")
