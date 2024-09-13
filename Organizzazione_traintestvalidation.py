#neutral in validation non riesce a riempiersi a causa dei pochi dati quindi si è riempito manualmente sottraendo un video per attore in neutral nel train
import os
import shutil
import random

# Percorso di partenza del dataset
source_dir = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/dataset organizzato'
# Percorso di destinazione per train, validation e test
dest_dir = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasettraintestvalidation'

# Creazione delle cartelle train, validation e test
train_dir = os.path.join(dest_dir, 'train')
validation_dir = os.path.join(dest_dir, 'validation')
test_dir = os.path.join(dest_dir, 'test')

# Assicurarsi che le cartelle di destinazione esistano
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Itera su ogni emozione
for emotion_folder in os.listdir(source_dir):
    emotion_path = os.path.join(source_dir, emotion_folder)
    if not os.path.isdir(emotion_path):
        continue

    # Itera su ogni attore
    for actor_folder in os.listdir(emotion_path):
        actor_path = os.path.join(emotion_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        # Elenco di tutti i file .mp4 dell'attore
        files = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
        total_files = len(files)

        if total_files == 0:
            continue

        # Mescola i file per una distribuzione casuale
        random.shuffle(files)
        
        # Determina le quantità per train, validation e test
        train_size = int(total_files * 0.8)
        remaining_size = total_files - train_size
        validation_size = int(remaining_size / 2)
        test_size = remaining_size - validation_size

        train_files = files[:train_size]
        validation_files = files[train_size:train_size + validation_size]
        test_files = files[train_size + validation_size:]

        # Funzione per copiare file evitando di creare cartelle vuote
        def copy_files(file_list, dest_subdir):
            if not file_list:
                return
            dest_subfolder = os.path.join(dest_subdir, emotion_folder, actor_folder)
            os.makedirs(dest_subfolder, exist_ok=True)
            for file in file_list:
                src_file = os.path.join(actor_path, file)
                dest_file = os.path.join(dest_subfolder, file)
                shutil.copy2(src_file, dest_file)

        # Copia dei file nelle rispettive cartelle
        copy_files(train_files, train_dir)
        copy_files(validation_files, validation_dir)
        copy_files(test_files, test_dir)
