import tensorflow as tf
from model.cnn_model import create_cnn_model
from data.data_loader import load_data_from_directory, augumentation
from pathlib import Path
import os



def train_and_save_model(model_save_path, log_dir, train_data, validation_data):

    model = create_cnn_model(input_shape=image_shape, output_shape=4)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss', patience = 10, restore_best_weights=True
    )

    model.fit(train_data, validation_data = validation_data, epochs = 20, callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb],verbose=2)
    model.save(model_save_path)

    print(f"model saved to {model_save_path}")


if __name__ == "__main__":
    train_path=Path("artifacts/data/ninjacart_data/train")
    model_path=Path("artifacts/model/model.keras")
    log_dir=Path("logs/ResNet")
    image_shape=(256,256,3)
    train_ds = load_data_from_directory(data_path=train_path, 
                                    label_mode="categorical",
                                    batch_size=32,
                                    image_size=(image_shape[0],image_shape[1]),
                                    subset="training",
                                    validation_split=0.2,
                                    shuffle=True)

    train_ds = augumentation(train_ds)

    validation_ds = load_data_from_directory(data_path=train_path,
                                         label_mode="categorical",
                                         batch_size=32,
                                         image_size=(image_shape[0],image_shape[1]),
                                         subset="validation",
                                         validation_split=0.2)
    train_and_save_model(model_save_path=model_path, log_dir=log_dir,train_data=train_ds, validation_data=validation_ds)

     

