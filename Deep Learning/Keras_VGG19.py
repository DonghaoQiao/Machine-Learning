from tensorflow.keras import models, layers

# VGG19: 16 conv layers + 3 fully connected layers
class VGG19(models.Sequential):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        # layer 1, conv layer
        self.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=input_shape, padding='same'))
        # layer 2, conv layer
        self.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=2, padding='valid'))
        # layer 3, conv layer
        self.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 4, conv layer
        self.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=2, padding='valid'))
        # layer 5, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 6, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 7, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 8, conv layer
        self.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=2, padding='valid'))
        # layer 9, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 10, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 11, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 12, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=2, padding='valid'))
        # layer 13, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 14, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 15, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # layer 16, conv layer
        self.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
        # pooling layer
        self.add(layers.MaxPooling2D(pool_size=2, padding='valid'))
        # flatten the CNN output so that we can connect it with fully connected layers
        self.add(layers.Flatten())
        # layer 17, fully connected layer
        self.add(layers.Dense(units=4096, activation='relu'))
        # add dropout to prevent overfitting
        self.add(layers.Dropout(0.4))
        # layer 18, fully connected layer
        self.add(layers.Dense(units=4096, activation='relu'))
        # add dropout to prevent overfitting
        self.add(layers.Dropout(0.4))
        # layer 19, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model = VGG19((224,224,3), 10)
model.summary()



