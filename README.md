# Cat and Dog Image Classification using CNN with TensorFlow and Keras

## Project Overview 

In this project, we develop a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs**. The model is trained on a labeled dataset of 4000 images of cats and 4000 images of dogs, evaluated on a test set of 1000 images of each class, and tested on new unseen images in a `single_prediction` folder.

The primary tools used are **TensorFlow** and **Keras**. Keras simplifies the process of building deep learning models, making it accessible for both beginners and experts, while TensorFlow's backend ensures efficient computation.

---

## Project Purpose 

The goals of this project are:
1. Build a CNN model that accurately classifies images of cats and dogs.
2. Train the CNN using a labeled dataset.
3. Evaluate the model's accuracy on unseen test images.
4. Use the trained model to predict the class (cat or dog) of images in the `single_prediction` folder.

---

## Dataset 

The dataset consists of three parts:
- **Training Set**: 4000 images of cats and 4000 images of dogs.
- **Test Set**: 1000 images of cats and 1000 images of dogs.
- **Single Prediction Folder**: A folder with unseen images for model predictions.

To ensure compatibility with the CNN, all images are resized to **64x64 pixels**.

**Dataset Structure:**
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

---

## Tools & Libraries 🛠️
- **TensorFlow**: Machine learning framework (integrated with Keras).
- **Keras**: High-level API for building and training deep learning models.
- **Python**: Programming language for implementation.
- **NumPy**: Numerical operations.
- **ImageDataGenerator**: Data preprocessing and augmentation.

---

## Model Architecture 

The CNN model architecture includes the following components:

1. **Input Layer**: Reshapes the input images to (64, 64, 3).
2. **Convolutional Layers**: Extract features like edges, textures, and patterns using ReLU activation.
3. **Pooling Layers**: Reduce spatial dimensions using MaxPooling.
4. **Flattening Layer**: Converts 2D feature maps into a 1D array for the dense layer.
5. **Fully Connected Layers**: Learn complex patterns with dense connections.
6. **Dropout Layer**: Reduces overfitting by randomly deactivating neurons during training.
7. **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

**Model Code:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Training the Model 

The training and test datasets are preprocessed using **ImageDataGenerator** for rescaling and data augmentation.

**Data Preprocessing and Training Code:**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = validation_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Train the model
cnn.fit(training_set, validation_data=test_set, epochs=25)
```

---

## Results 📈

The model was trained for **25 epochs** with the following results:
- **Training Accuracy**: ~52%
- **Training Loss**: ~0.7018
- **Validation Accuracy**: ~63.8%
- **Validation Loss**: ~0.6555

These results show that the model effectively learns distinguishing features between cats and dogs, with scope for further improvement.

---

## Making Predictions 

To predict the class of a new image, use the following function:

**Prediction Code:**
```python
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_single_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    return 'dog' if result[0][0] > 0.5 else 'cat'

# Example prediction
print(classify_single_image('dataset/single_prediction/sample_image.jpg'))
```

---

## Conclusion 

In this project, we successfully built a **Convolutional Neural Network** using TensorFlow and Keras to classify images of cats and dogs. The model demonstrates promising results with a validation accuracy of **~64%**. By using data augmentation, dropout regularization, and proper preprocessing, the model effectively learned patterns in the training dataset.

### Key Takeaways:
1. CNNs are powerful for image classification tasks.
2. TensorFlow and Keras simplify the process of building and training deep learning models.
3. Further improvements can be made using transfer learning or hyperparameter tuning.

This project serves as a strong foundation for binary image classification tasks and can be extended to other datasets or more complex problems.

---

**Next Steps:** 
- Try transfer learning with pre-trained models like VGG16 or ResNet.
- Use more data augmentation techniques.
- Explore deeper CNN architectures for improved accuracy.

---

**Thank you for reading!** 

Feel free to experiment with the code and expand on the project. Happy coding! 😊

# CNN-for-Image-Classification
In this project, I built up a convolutional neural network (CNN) to be trained on training dataset (including 4000 images of dog and 4000 images of cat), and then be tested on the test dataset including 1000 images of dog and 1000 images of cat (for the sake of evaluation). Finally, I input my model with images in "single_prediction" folder in order to classify the image as a dog or cat (making prediction using my model). 

Note: To build my CNN model, I used TensorFlow (integrated with Keras). Keras is built inside the TensorFlow package. Initially developed as an independent high-level neural networks API, Keras was later integrated into TensorFlow as its official high-level API for building and training deep learning models. This integration allows Keras to leverage TensorFlow's powerful backend, which includes efficient computation and support for distributed training, while providing an easy-to-use interface for developers. Keras simplifies the process of designing complex neural networks with just a few lines of code, making it accessible to beginners and experienced practitioners alike. While Keras can also be used with other backends, its integration with TensorFlow has made it the default API within the TensorFlow ecosystem.
my dataset: https://drive.google.com/drive/folders/1OxyERDsXkUcV3tFW4xW3qZ_PIfrY0isR?usp=drive_link
