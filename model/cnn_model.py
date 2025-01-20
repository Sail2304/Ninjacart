import tensorflow as tf

def create_cnn_model(input_shape=(256,256,3), output_shape=4):

    base_model_1 = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top = False)
    model= tf.keras.Sequential([
            #Normalizing 0-255 into 0 to 1
            tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
            base_model_1,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(rate = 0.1),
            tf.keras.layers.Dense(units=output_shape, activation = 'softmax')
        ])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['accuracy', 'Precision', 'Recall'])
    
    return model
