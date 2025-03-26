import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def simple_cnn(input_shape, num_classes):

    model = keras.Sequential([
        layers.Input(shape = input_shape),
        layers.Conv2D(filters = 10, kernel_size = (3, 3), activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(num_classes, activation = 'softmax')
    ]
    )

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model
    

input_shape = (15, 15, 1)
num_classes = 10

model = simple_cnn(input_shape, num_classes)
print(model.summary())

