# CNN-Image-Classification-CODTECH
# Image Classification with CNN
## CODTECH IT Solutions - Internship Task 3


## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for image classification using **TensorFlow/Keras**. The model is trained on the **CIFAR-10 dataset**, which contains 60,000 32x32 color images across 10 different classes.

**Internship:** CODTECH IT Solutions  
**Task:** Task 3 - Image Classification Model  
**Objective:** Build a functional CNN model with performance evaluation on test dataset  

---

## 🎯 Objectives

- Design and implement a CNN architecture for image classification
- Train the model on CIFAR-10 dataset with data augmentation
- Evaluate model performance using comprehensive metrics
- Visualize predictions and analyze model behavior
- Achieve high accuracy on test dataset

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.8+**

### Deep Learning Framework
- **TensorFlow 2.x**
- **Keras API**

### Libraries
- **Data Processing:** numpy, pandas
- **Visualization:** matplotlib, seaborn
- **Model Evaluation:** scikit-learn

---

## 📂 Project Structure

```
image-classification-cnn/
│
├── image_classification_cnn.ipynb    # Main Jupyter notebook
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── SETUP.md                           # Setup instructions
│
├── models/                            # (Generated during training)
│   ├── best_cnn_model.h5             # Best model checkpoint
│   ├── cifar10_cnn_model.h5          # Final trained model
│   ├── model_architecture.json       # Model architecture
│   └── model_weights.h5              # Model weights
│
└── results/                           # (Generated during evaluation)
    ├── training_history.png          # Training curves
    ├── confusion_matrix.png          # Confusion matrix
    └── predictions.png               # Sample predictions
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-classification-cnn.git
cd image-classification-cnn
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook image_classification_cnn.ipynb
```

---

## 📊 Dataset

### CIFAR-10 Dataset
- **Size:** 60,000 images (50,000 training + 10,000 testing)
- **Image Dimensions:** 32x32 pixels (RGB)
- **Classes:** 10 categories
  1. Airplane
  2. Automobile
  3. Bird
  4. Cat
  5. Deer
  6. Dog
  7. Frog
  8. Horse
  9. Ship
  10. Truck

The dataset is automatically downloaded when running the notebook for the first time.

---

## 🏗️ CNN Model Architecture

### Architecture Overview
```
Input (32x32x3)
    ↓
[Convolutional Block 1]
├── Conv2D (32 filters, 3x3) + ReLU
├── Batch Normalization
├── Conv2D (32 filters, 3x3) + ReLU
├── Batch Normalization
├── MaxPooling2D (2x2)
└── Dropout (0.25)
    ↓
[Convolutional Block 2]
├── Conv2D (64 filters, 3x3) + ReLU
├── Batch Normalization
├── Conv2D (64 filters, 3x3) + ReLU
├── Batch Normalization
├── MaxPooling2D (2x2)
└── Dropout (0.25)
    ↓
[Convolutional Block 3]
├── Conv2D (128 filters, 3x3) + ReLU
├── Batch Normalization
├── Conv2D (128 filters, 3x3) + ReLU
├── Batch Normalization
├── MaxPooling2D (2x2)
└── Dropout (0.25)
    ↓
[Dense Layers]
├── Flatten
├── Dense (256) + ReLU
├── Batch Normalization
├── Dropout (0.5)
├── Dense (128) + ReLU
├── Batch Normalization
├── Dropout (0.5)
└── Dense (10) + Softmax
    ↓
Output (10 classes)
```

### Key Features
- **3 Convolutional Blocks** with increasing filter sizes (32 → 64 → 128)
- **Batch Normalization** for stable and faster training
- **Dropout Layers** to prevent overfitting
- **MaxPooling** for dimensionality reduction
- **Dense Layers** for final classification

---

## 🔍 Methodology

### 1. **Data Preprocessing**
   - Normalize pixel values to [0, 1] range
   - One-hot encode labels
   - Split data: 80% training, 20% validation

### 2. **Data Augmentation**
   - Random rotation (±15 degrees)
   - Random width/height shift (±10%)
   - Random horizontal flip
   - Random zoom (±10%)

### 3. **Model Training**
   - Optimizer: Adam (learning rate: 0.001)
   - Loss Function: Categorical Crossentropy
   - Batch Size: 64
   - Maximum Epochs: 50
   
### 4. **Training Callbacks**
   - **Early Stopping:** Stops training if validation loss doesn't improve (patience=10)
   - **Learning Rate Reduction:** Reduces LR when validation loss plateaus (factor=0.5, patience=5)
   - **Model Checkpoint:** Saves best model based on validation accuracy

### 5. **Model Evaluation**
   - Test accuracy and loss
   - Confusion matrix
   - Per-class accuracy
   - Classification report (precision, recall, F1-score)
   - Prediction confidence analysis

---

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~70-85% |
| **Test Top-5 Accuracy** | ~95-98% |
| **Test Loss** | ~0.5-0.8 |
| **Training Time** | ~10-20 minutes (CPU) / ~3-5 minutes (GPU) |

*Note: Actual results may vary based on training configuration and hardware*

### Best Performing Classes
Typically achieves highest accuracy on:
- Ship (~85-90%)
- Automobile (~80-85%)
- Truck (~75-85%)

### Challenging Classes
Lower accuracy on:
- Cat (~60-70%)
- Bird (~65-75%)
- Dog (~60-70%)

---

## 💡 Key Features

✅ **Complete CNN Pipeline**
- Data loading and preprocessing
- Model architecture design
- Training with callbacks
- Comprehensive evaluation

✅ **Data Augmentation**
- Improves model generalization
- Prevents overfitting
- Increases effective dataset size

✅ **Advanced Visualizations**
- Training history plots
- Confusion matrix heatmap
- Per-class accuracy charts
- Correct and incorrect predictions
- Confidence distribution analysis

✅ **Model Persistence**
- Save/load trained models
- Export model architecture
- Save model weights separately

✅ **Interactive Predictions**
- Predict on individual images
- Display confidence scores
- Visualize prediction probabilities

---

## 🎨 Visualizations

The notebook includes:

1. **Sample Images** - Visual representation of each class
2. **Class Distribution** - Bar charts showing data balance
3. **Data Augmentation Examples** - Before/after augmentation
4. **Training History** - Accuracy and loss curves
5. **Confusion Matrix** - Model prediction patterns
6. **Per-Class Accuracy** - Performance breakdown by class
7. **Prediction Samples** - Correct and incorrect predictions
8. **Confidence Analysis** - Distribution of prediction confidence

---

## 📝 Usage Example

```python
# Load the trained model
from tensorflow import keras
model = keras.models.load_model('cifar10_cnn_model.h5')

# Make prediction on a single image
import numpy as np
image = X_test[0]  # Get test image
image_normalized = image.astype('float32') / 255.0
prediction = model.predict(np.expand_dims(image_normalized, axis=0))
predicted_class = np.argmax(prediction)

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {prediction[0][predicted_class]*100:.2f}%")
```

---

## 🔮 Future Enhancements

- [ ] Implement transfer learning with pre-trained models (ResNet50, VGG16, EfficientNet)
- [ ] Add more sophisticated data augmentation (CutMix, MixUp)
- [ ] Experiment with deeper architectures
- [ ] Implement learning rate schedulers (Cosine Annealing, Step Decay)
- [ ] Add Grad-CAM for model interpretability
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create web interface for real-time predictions
- [ ] Extend to other datasets (CIFAR-100, ImageNet subset)

---

## 📚 Learning Outcomes

Through this project, I learned:

✔️ CNN architecture design principles  
✔️ Image preprocessing and augmentation techniques  
✔️ TensorFlow/Keras model building and training  
✔️ Regularization techniques (Dropout, Batch Normalization)  
✔️ Training optimization with callbacks  
✔️ Model evaluation and performance analysis  
✔️ Visualization of deep learning results  
✔️ Model persistence and deployment preparation  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## 🙏 Acknowledgments

- **CODTECH IT Solutions** for the internship opportunity
- **TensorFlow/Keras** team for the excellent deep learning framework
- **CIFAR-10** dataset creators
- Open-source community for valuable resources

---

## 📞 Contact

For any queries or feedback, please reach out:

- **Email:** whoamritasharma@gmail.com

---

## 📖 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CNN Architectures](https://arxiv.org/abs/1409.1556)

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Last Updated:** February 2026  
**Status:** ✅ Completed
