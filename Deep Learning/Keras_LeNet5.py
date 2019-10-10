from tensorflow.keras import models, layers
import tensorflow.keras as keras

class LeNet(models.Sequential):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        # layer 1, conv layer
        self.add(layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=input_shape, padding='same'))
        # layer 2, pooling layer
        self.add(layers.AveragePooling2D(pool_size=2, strides=1, padding='valid'))
        # layer 3, conv layer
        self.add(layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid'))
        # layer 4, pooling layer
        self.add(layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'))
        # layer 5, fully connected conv layer
        self.add(layers.Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh', padding='valid'))
        #Flatten the CNN output so that we can connect it with fully connected layers
        self.add(layers.Flatten())
        # layer 6, fully connected layer
        self.add(layers.Dense(units=84, activation='tanh'))
        # layer 7, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])