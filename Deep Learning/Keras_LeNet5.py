from tensorflow.keras import models, layers

# 2 conv layers + 3 fully connected layers
class LeNet(models.Sequential):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        # layer 1, conv layer
        self.add(layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='sigmoid', input_shape=input_shape, padding='same'))
        # pooling layer
        self.add(layers.AveragePooling2D(pool_size=2, strides=1, padding='valid'))
        # layer 2, conv layer
        self.add(layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='sigmoid', padding='valid'))
        # pooling layer
        self.add(layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'))
        # flatten the CNN output so that we can connect it with fully connected layers
        self.add(layers.Flatten())
        # layer 3, fully connected layer
        self.add(layers.Dense(units=120, activation='sigmoid'))
        # layer 4, fully connected layer
        self.add(layers.Dense(units=84, activation='sigmoid'))
        # layer 5, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model = LeNet((224,224,3), 10)
model.summary()