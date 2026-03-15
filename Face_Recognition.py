from google.colab import files
uploaded = files.upload()  # A file selector will pop up

!ls
# Install required packages
%pip install --quiet torch torchvision facenet-pytorch Pillow

# Restart runtime automatically to avoid import issues


# import os
# os.kill(os.getpid(), 9)
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
def get_embedding(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)  # detect & align
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0))
        return embedding[0].numpy()
    else:
        return None

emb1 = get_embedding("srk1.jpg")
emb2 = get_embedding("srk2.jpg")

if emb1 is not None and emb2 is not None:
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(" Cosine Similarity:", cos_sim)
else:
    print(" Could not detect face in one of the images.")


# emb1 and emb2 are Tensors (not numpy)
emb1_tensor = torch.tensor(emb1)
emb2_tensor = torch.tensor(emb2)

cos_sim = torch.nn.functional.cosine_similarity(emb1_tensor.unsqueeze(0), emb2_tensor.unsqueeze(0))
print("Cosine Similarity:", cos_sim.item())

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Example: true labels (1=same person, 0=different)
y_true = [1, 1, 0, 0, 1, 0, 1, 0]

cos_sims = [0.82, 0.75, 0.60, 0.40, 0.79, 0.55, 0.85, 0.30]

threshold = 0.8
y_pred = [1 if sim >= threshold else 0 for sim in cos_sims]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
##