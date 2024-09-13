import os
import cv2

def create_directory_structure(src, dst):
    for root, dirs, files in os.walk(src):
        for directory in dirs:
            src_dir_path = os.path.join(root, directory)
            dst_dir_path = src_dir_path.replace(src, dst, 1)
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)

def extract_frames_from_videos(src, dst, fps=10):
    create_directory_structure(src, dst)
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                dst_dir = root.replace(src, dst, 1)
                video_name = os.path.splitext(file)[0]
                
                cap = cv2.VideoCapture(video_path)
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(frame_rate / fps)
                
                frame_count = 0
                success, frame = cap.read()
                
                while success:
                    if frame_count % frame_interval == 0:
                        frame_number = frame_count // frame_interval
                        frame_filename = f"{video_name}-{frame_number:04d}.jpg"
                        frame_path = os.path.join(dst_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                    frame_count += 1
                    success, frame = cap.read()
                
                cap.release()

src_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasettraintestvalidation'
dst_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetframe'

extract_frames_from_videos(src_path, dst_path, fps=10)
