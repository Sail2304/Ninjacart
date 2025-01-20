from utils.preprocessing import preprocess_image
import tensorflow as tf
import numpy as np

def load_model(model_path):
    """ 
    load pretrained CNN model

    """
    model = tf.keras.models.load_model(model_path)

    return model

def predict_image(model, img):
    """
    Given a preprocessed image, predicts its class using the loaded model.
    """
    class_names = ['Noise', 'Onion', 'Potato', 'Tomato']
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    print(prediction)
    predicted_class = tf.argmax(prediction[0])
    return class_names[predicted_class], prediction[0][predicted_class]

