import tensorflow as tf

def preprocess_image(img, target_size=(256,256)):
        img = tf.keras.utils.load_img(img)
        img = tf.keras.utils.img_to_array(img)
        img = tf.image.resize(img, target_size)
        img = tf.expand_dims(img, axis = 0)

        return img