# Sign Language Detection 🧠🤟

This project implements a deep learning model using **InceptionV3** to detect hand gestures representing sign language alphabets (A–Z and 0–9). The model is trained on image datasets using TensorFlow and Keras, with data augmentation to improve generalization.

---

## 🗂 Dataset Preparation

- Images are organized by class (A–Z, 0–9) into folders.
- `ImageDataGenerator` is used for data augmentation and preprocessing:
  - Rescaling
  - Zoom
  - Rotation
  - Shear

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30
)
val_datagen = ImageDataGenerator(rescale=1./255)
```

---

## 🏗 Model Architecture

- Base Model: **InceptionV3** pretrained on ImageNet
- Final layers are replaced with a `Dense` layer using softmax for classification.

```python
from keras.applications import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

- Custom top layers are added and compiled with `adam` optimizer and `categorical_crossentropy` loss.

---

```python
model.fit(train_generator, epochs=5, validation_data=validation_generator)
```

---

## 💾 Saving Model

- The final trained model is saved for deployment or further use:

```python
model.save('sign_language_model.h5')
```

## 📌 Requirements

- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Matplotlib

Install them using:

```bash
pip install tensorflow keras numpy matplotlib
``'
-end