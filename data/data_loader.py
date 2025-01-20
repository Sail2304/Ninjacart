import tensorflow as tf


def load_data_from_directory(data_path, label_mode, batch_size, image_size, subset, validation_split=0, shuffle=False):

    ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_path,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size,
        subset=subset,
        validation_split=validation_split,
        shuffle=shuffle,
        seed=2022
    )

    return ds


def augumentation(data):
    augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomTranslation(height_factor = 0.2, width_factor=0.2)
    ])

    aug_ds = data.map(lambda x, y: (augmentation(x), y))
    
    return aug_ds
