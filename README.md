# Cat and Dog Image Classification using CNN with TensorFlow and Keras


## 1. Installations
This project was written in Python, using Jupyter Notebook. The relevant Python libraries for this project are as follows:

- **TensorFlow**: `pip install tensorflow`
- **Keras**: Built into TensorFlow
- **NumPy**: `pip install numpy`
- **ImageDataGenerator**: Part of TensorFlow/Keras for data preprocessing
- **Matplotlib** (optional for visualization): `pip install matplotlib`

---

## 2. Project Motivation
This project focuses on building a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs** using TensorFlow (integrated with Keras). 

The goals of the project are:
1. Build a CNN model that can distinguish between images of cats and dogs.
2. Train the model on a labeled dataset of cats and dogs.
3. Evaluate the model's performance on unseen test images.
4. Make predictions on new images stored in the `single_prediction` folder.

By successfully implementing this project, we gain insights into:
- Image preprocessing and augmentation using `ImageDataGenerator`.
- Building, training, and evaluating a CNN.
- Making predictions with a trained deep learning model.

---

## 3. Dataset
The dataset consists of labeled images of cats and dogs, split into three sections:
- **Training Set**: 4000 images of cats and 4000 images of dogs.
- **Test Set**: 1000 images of cats and 1000 images of dogs.
- **Single Prediction Folder**: Contains unseen images for testing predictions.

**Dataset Structure**:
```
dataset/
│
├── training_set/
│   ├── cats/
│   └── dogs/
│
├── test_set/
│   ├── cats/
│   └── dogs/
│
└── single_prediction/
    ├── sample_image1.jpg
    └── sample_image2.jpg
```

All images are resized to **64x64 pixels** to match the input requirements of the CNN.

my dataset google drive link:

https://drive.google.com/drive/folders/1OxyERDsXkUcV3tFW4xW3qZ_PIfrY0isR?usp=drive_link
---

## 4. File Descriptions
This project contains the following files:

1. **CNN_Image_Classification.ipynb**: Main Jupyter Notebook implementing the entire workflow, including:
   - Dataset preprocessing
   - CNN model architecture
   - Model training and evaluation
   - Predictions on unseen images

2. **Dataset Folder**: Contains the images for training, testing, and predictions.

3. **README.md**: Documentation of the project and its workflow.

---

## 5. CNN Model Architecture
The CNN architecture includes the following layers:

1. **Input Layer**: Resizes images to (64, 64, 3).
2. **Convolutional Layer 1**: Extracts features using 32 filters (3x3 kernel) and ReLU activation.
3. **MaxPooling Layer 1**: Reduces spatial dimensions (2x2 pool size).
4. **Convolutional Layer 2**: Another 32 filters with ReLU activation.
5. **MaxPooling Layer 2**: Reduces spatial dimensions again.
6. **Flatten Layer**: Converts feature maps to a 1D array.
7. **Dense Layer**: Fully connected layer with 128 neurons and ReLU activation.
8. **Dropout Layer**: Reduces overfitting by deactivating 50% of neurons during training.
9. **Output Layer**: A single neuron with a sigmoid activation for binary classification.

---

## 6. Results

The main findings of the code can be found at the post on Medium available : 
https://medium.com/@hanyue856/building-a-cnn-for-cat-and-dog-image-classification-with-tensorflow-and-keras-ac028f6b5927
```
```
---

## 7. Acknowledgements

Thanks to TensorFlow and Keras for providing robust tools for deep learning.

---

Thank you for exploring this project!  Feel free to use or extend the code for your own experiments and applications.


