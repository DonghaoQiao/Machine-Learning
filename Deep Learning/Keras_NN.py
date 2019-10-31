import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras import models, layers

class NN(models.Sequential):
    def __init__(self, input_shape, n_classes, hidden_nodes=200):
        super().__init__()
        # layer 1, fully connected layer
        self.add(layers.Dense(units=hidden_nodes,activation='sigmoid',input_shape=input_shape))
        # layer 2, output layer
        self.add(layers.Dense(units=n_classes, activation='softmax'))
        
        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

from loadMNIST import *
train_x,train_y,test_x,test_y=load_norm_mnist()
train_x=train_x_raw.reshape(60000,28*28)
test_x=test_x_raw.reshape(10000,28*28)


model = NN((train_x.shape[1:]), 10)
print(model.summary())

hist = model.fit(train_x, train_y, batch_size=50, epochs=10, validation_split=0.2)

print(model.evaluate(test_x, test_y))
plt.plot(hist.history['acc'], color = 'red')
plt.plot(hist.history['val_acc'], color = 'blue')

pred_y=np.argmax(model.predict(test_x),axis=1)
print(pred_y)