
# Face Recognition Using Inception-ResNet-V1

This repository contains a **Face Verification System** built with PyTorch. The system detects faces, extracts high-dimensional embeddings, and calculates the similarity between images to identify if they belong to the same person.

---

## 🚀 Key Features

* **Face Detection:** Uses **MTCNN** to detect, crop, and align faces.
* **Feature Extraction:** Leverages a pre-trained **Inception-ResNet-V1** (FaceNet) model.
* **Similarity Analysis:** Measures **Cosine Similarity** between image pairs.
* **Performance Evaluation:** Calculates Accuracy, Precision, Recall, and F1-Score.

---

## 🛠️ Installation

To run this project locally, install the following dependencies:

```bash
pip install torch torchvision facenet-pytorch Pillow numpy scikit-learn

```

---

## 📖 How to Use

1. **Prepare Images:** Place your images (e.g., `image1.jpg`, `image2.jpg`) in the project directory.
2. **Run the Script:**
```python
python Face_Recognition_Using_Inception_ResNet.py

```


3. **Check Results:** The script will output a **Cosine Similarity** score. A score closer to **1.0** indicates the same person.

---

## 📊 Evaluation Results

The model uses a default threshold of **0.8** to classify matches.

* **True Positives (TP):** Correctly identified matches.
* **Accuracy:** Overall correctness of the model.
* **F1-Score:** The balance between Precision and Recall.

---

**Author:** [AkhilT9](https://github.com/AkhilT9)

---

