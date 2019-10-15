from tensorflow.keras import models, layers

# 5 conv layers + 3 fully connected layers
class AlexNet(models.Sequential):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        # layer 1, conv layer
        self.add(layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape, padding='valid'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'))
        # layer 2, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'))
        # layer 3, conv layer
        self.add(layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 4, conv layer
        self.add(layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 5, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'))
        # flatten the CNN output so that we can connect it with fully connected layers
        self.add(layers.Flatten())
        # layer 6, fully connected layer
        self.add(layers.Dense(units=4096, activation='relu'))
        # add dropout to prevent overfitting
        self.add(layers.Dropout(0.4))
        # layer 7, fully connected layer
        self.add(layers.Dense(units=4096, activation='relu'))
        # add dropout to prevent overfitting
        self.add(layers.Dropout(0.4))
        # layer 8, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model = AlexNet((224,224,3), 10)
model.summary()



