import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "dataset"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage)
        Id = int(os.path.split(imagePaths)[-1].split(".")[1])  # Convert label to integer
        faces.append(faceNP)
        ids.append(Id)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)

    return ids, faces

IDs, facedata = getImageID(path)

# Convert IDs to numpy array with integer type
IDs = np.array(IDs, dtype=np.int32)

recognizer.train(facedata, IDs)
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed...")
