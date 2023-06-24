from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

learn_rate = 1e-5
epochs = 20
batch_size = 32

dataset_directory = r"dataset"
sub_categories = ["with_mask", "without_mask"]

print("\n[STATUS] - Started Image Data Loading from Directory...\n")

data = []
labels = []

for category in sub_categories:
    path = os.path.join(dataset_directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(226, 226))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, random_state=42)

image_augment = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(226, 226, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("\n[STATUS] - Model Compilation...")
optimizer = Adam(lr=learn_rate, decay=learn_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("\n[STATUS] - Training...\n")
mod = model.fit(image_augment.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size, validation_data=(X_test, y_test), validation_steps=len(X_test) // batch_size, epochs=epochs)

print("\n[STATUS] - Network Evaluation...\n")
predict = model.predict(X_test, batch_size=batch_size)
predict = np.argmax(predict, axis=1)

print(classification_report(y_test.argmax(axis=1), predict, target_names=lb.classes_))

print("\n[STATUS] - Saving Trained Model to face_mask_detector.model")
model.save("face_mask_detector.model", save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), mod.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), mod.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), mod.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), mod.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")