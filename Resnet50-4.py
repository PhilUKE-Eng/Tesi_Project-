import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from Dataloader3 import create_dataloaders  

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

# Percorso al dataset
base_dir = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetcrop"

# Creazione dei dataloader e calcolo del numero di classi
print("Creazione dei dataloader...")
train_loader, val_loader, test_loader, num_classes = create_dataloaders(base_dir, batch_size=32)
print(f"Numero di classi: {num_classes}")
print("Dataloader creati con successo.")

# Caricamento del modello ResNet-50 con Dropout
print("Caricamento del modello ResNet-50 con Dropout...")
class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithDropout, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Aggiungi Dropout con probabilità di 0.5
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = ResNet50WithDropout(num_classes).to(device)
print("Modello caricato con successo.")

# Impostazione dell'ottimizzatore e della funzione di perdita
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Funzione di allenamento
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(data_loader, desc="Training Epoch", unit="batch"):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Funzione di validazione
def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation Epoch", unit="batch"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy, all_labels, all_preds

# Funzione per salvare la matrice di confusione
def save_confusion_matrix(labels, preds, num_classes, file_path):
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()

# Funzione per salvare le previsioni e le etichette su file
def save_predictions_and_labels(labels, preds, file_path):
    with open(file_path, 'w') as f:
        for label, pred in zip(labels, preds):
            f.write(f"Label: {label}, Prediction: {pred}\n")

# Creazione della directory per salvare i modelli e i report se non esiste
save_dir = "/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Modelli/Modello_res50crop/train2"
os.makedirs(save_dir, exist_ok=True)

# Impostazione per Early Stopping
patience = 2  # Numero di epoche senza miglioramenti prima di interrompere
best_val_loss = float('inf')
epochs_without_improvement = 0

# Esecuzione dell'allenamento
num_epochs = 20  
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Esegui un'epoca di allenamento
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    
    # Esegui un'epoca di validazione
    val_loss, val_accuracy, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Salvataggio del modello se migliora
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        model_save_path = os.path.join(save_dir, f'modelresnet50_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Modello migliorato salvato in {model_save_path}")
    else:
        epochs_without_improvement += 1
        print(f"Nessun miglioramento. Epochs senza miglioramenti: {epochs_without_improvement}/{patience}")

    # Salvataggio della matrice di confusione e dei report
    cm_save_path = os.path.join(save_dir, f'confusion_matrix_epoch{epoch+1}.png')
    save_confusion_matrix(val_labels, val_preds, num_classes, cm_save_path)
    print(f"Matrice di confusione salvata in {cm_save_path}")

    report_save_path = os.path.join(save_dir, f'classification_report_epoch{epoch+1}.txt')
    with open(report_save_path, 'w') as f:
        report = classification_report(val_labels, val_preds, labels=list(range(num_classes)))
        f.write(report)
    print(f"Report di classificazione salvato in {report_save_path}")
    
    # Salvataggio dei valori numerici delle uscite con le etichette
    predictions_save_path = os.path.join(save_dir, f'predictions_epoch{epoch+1}.txt')
    save_predictions_and_labels(val_labels, val_preds, predictions_save_path)
    print(f"Valori numerici delle uscite salvati in {predictions_save_path}")

    # Aggiornamento del learning rate
    scheduler.step(val_loss)

    # Interruzione anticipata se non ci sono miglioramenti significativi
    if epochs_without_improvement >= patience:
        print("Early stopping attivato. Interruzione dell'addestramento.")
        break

print("Training completato.")
