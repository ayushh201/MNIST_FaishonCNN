# MNIST Fashion Classification Project Using CNN  

This project demonstrates a fashion classification system using the MNIST Fashion dataset with Machine Learning (ML) and a Convolutional Neural Network (CNN). The key steps include data visualization, model training using CNN, and evaluation using a confusion matrix.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Technologies Used](#technologies-used)  
3. [Dataset Description](#dataset-description)  
4. [Project Workflow](#project-workflow)  
5. [Model Evaluation](#model-evaluation)  
6. [Conclusion](#conclusion)  

---

## **Project Overview**  
The goal of this project is to classify images from the MNIST Fashion dataset using a CNN model. The dataset consists of grayscale images of clothing items, categorized into 10 classes such as T-shirts, pants, shoes, and bags.  

---

## **Technologies Used**  
- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## **Dataset Description**  
The MNIST Fashion dataset includes:  
- **60,000 training images**  
- **10,000 test images**  
- **10 Classes:**  
  - 0 => T-shirt/top  
  - 1 => Trouser  
  - 2 => Pullover  
  - 3 => Dress  
  - 4 => Coat  
  - 5 => Sandal  
  - 6 => Shirt  
  - 7 => Sneaker  
  - 8 => Bag  
  - 9 => Ankle boot  

---

## **Project Workflow**  

### 1. **Importing Libraries**  
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
```

---

### 2. **Loading and Visualizing Data**  
```python
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display sample images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[i]])
plt.show()
```

---

### 3. **Building the CNN Model**  
```python
# Building the CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

### 4. **Training the Model**  
```python
# Reshape the images for CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```

---

### 5. **Model Evaluation**  
```python
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

# Generate confusion matrix
predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, pred_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(test_labels, pred_labels, target_names=class_names))
```

---

## **Conclusion**  
The CNN model effectively classifies fashion items from the MNIST Fashion dataset using convolutional and pooling layers with an accuracy of 92%. The model learns image features, enabling better classification performance.
