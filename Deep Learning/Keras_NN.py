from tensorflow.keras import models, layers

class NN(models.Sequential):
    def __init__(self, input_shape, n_classes, hidden_nodes=200):
        super().__init__()
        # layer 1, fully connected layer
        self.add(layers.Dense(units=hidden_nodes,activation='sigmoid',input_shape=input_shape))
        # layer 2, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))
        
        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model = NN(([28*28]), 10)
print(model.summary())
