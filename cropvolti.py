import os
import cv2
from ultralytics import YOLO

def create_directory_structure(src, dst):
    for root, dirs, files in os.walk(src):
        for directory in dirs:
            src_dir_path = os.path.join(root, directory)
            dst_dir_path = src_dir_path.replace(src, dst, 1)
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)

def extract_faces_from_frames(src, dst, yolo_model):
    create_directory_structure(src, dst)
    model = YOLO(yolo_model)
    
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".jpg"):
                frame_path = os.path.join(root, file)
                dst_dir = root.replace(src, dst, 1)
                frame_name = os.path.splitext(file)[0]
                
                frame = cv2.imread(frame_path)
                
                # Detect faces and save cropped faces
                results = model(frame)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        face_crop = frame[y1:y2, x1:x2]
                        crop_filename = f"{frame_name}.jpg"
                        crop_path = os.path.join(dst_dir, crop_filename)
                        cv2.imwrite(crop_path, face_crop)

src_frames_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetframe'
dst_crop_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Dataset/datasetcrop'
yolo_model_path = '/home/filippo/Progetto_Tesi/MaterialeTesi/Progetto/ProgettoTesi3.0/Script/yolov8n-face.pt'

extract_faces_from_frames(src_frames_path, dst_crop_path, yolo_model_path)
