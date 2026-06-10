import tensorflow as tf


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (299, 299))
    return tf.keras.applications.inception_v3.preprocess_input(img)