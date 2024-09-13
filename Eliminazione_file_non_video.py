import os
import glob

# Definisci il percorso principale
base_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/Video_Song_Actor'

# Scorri tutte le cartelle dell'attore
for actor_folder in os.listdir(base_path):
    actor_path = os.path.join(base_path, actor_folder)
    
    if os.path.isdir(actor_path):
        # Scorri tutti i file mp4 nella cartella dell'attore
        for mp4_file in glob.glob(os.path.join(actor_path, '*.mp4')):
            # Estrai il nome del file senza estensione
            file_name = os.path.basename(mp4_file)
            name_parts = file_name.split('-')
            
            # Controlla se il nome del file corrisponde ai criteri
            if len(name_parts) > 2 and (name_parts[0] == '01' or name_parts[0] == '03'):
                # Cancella il file
                os.remove(mp4_file)
                print(f"File rimosso: {mp4_file}")
