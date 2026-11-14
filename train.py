"""Member 2 - MobileNetV2 transfer-learning classifier (benign vs malignant), TensorFlow/Keras."""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODEL_PATH = os.path.join(BASE, "dermassist_model.keras")
IMG, BATCH = 224, 32
EPOCHS_HEAD, EPOCHS_FINE = 6, 6

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA, "train"), image_size=(IMG, IMG), batch_size=BATCH,
    label_mode="binary", shuffle=True, seed=42)
val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA, "val"), image_size=(IMG, IMG), batch_size=BATCH,
    label_mode="binary", shuffle=False)
print("Classes:", train_ds.class_names, "(sigmoid output = P(malignant))")

# class weights to handle imbalance
counts = [len(os.listdir(os.path.join(DATA, "train", c))) for c in train_ds.class_names]
total = sum(counts)
class_weight = {i: total / (len(counts) * n) for i, n in enumerate(counts)}
print("Train counts:", dict(zip(train_ds.class_names, counts)), "weights:", class_weight)

augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False,
                                      weights="imagenet", name="base")
base.trainable = False
inp = keras.Input((IMG, IMG, 3))
x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(inp)   # MobileNetV2 preprocessing
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="drop")(x)
out = layers.Dense(1, activation="sigmoid", name="out")(x)
model = keras.Model(inp, out)


def compile_model(lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Recall(name="recall")])


cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3,
                                    restore_best_weights=True)]

print("\n--- Phase 1: train classification head ---")
compile_model(1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
          class_weight=class_weight, callbacks=cb, verbose=2)

print("\n--- Phase 2: fine-tune top layers ---")
base.trainable = True
for l in base.layers[:-30]:
    l.trainable = False
for l in base.layers:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
compile_model(1e-5)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
          class_weight=class_weight, callbacks=cb, verbose=2)

model.save(MODEL_PATH)
print("Saved model ->", MODEL_PATH)

# final evaluation
y_true = np.concatenate([y.numpy() for _, y in val_ds]).ravel()
y_prob = model.predict(val_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
