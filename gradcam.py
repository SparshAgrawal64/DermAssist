"""Member 3 - Grad-CAM explainability for the Keras model built in train.py."""
import cv2
import numpy as np
import tensorflow as tf


def gradcam_overlay(model, img_rgb, size=224, alpha=0.4):
    """img_rgb: uint8 RGB image. Returns (overlay_rgb, P(malignant))."""
    small = cv2.resize(img_rgb, (size, size)).astype("float32")
    x = model.get_layer("rescale")(tf.convert_to_tensor(small[None]))
    base, gap = model.get_layer("base"), model.get_layer("gap")
    drop, out = model.get_layer("drop"), model.get_layer("out")

    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        pred = out(drop(gap(conv), training=False))[:, 0]
    grads = tape.gradient(pred, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)
    return overlay, float(pred.numpy()[0])
