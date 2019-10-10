import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

# load dataset
(train_x_raw,train_y_raw),(test_x_raw,test_y_raw)=keras.datasets.mnist.load_data()
def load_mnist():
	# normalization
	train_x = train_x_raw/255
	test_x = test_x_raw/255
	train_y = keras.utils.to_categorical(train_y_raw, 10)
	test_y = keras.utils.to_categorical(test_y_raw, 10)
	return(train_x,train_y,test_x,test_y)

def plot_mnist(i):
	# plot an exampe
	plt.imshow(train_x_raw[i], cmap='gray')
	plt.title(train_y_raw[i])
	plt.show()
    
def plot_miss(pred_y):
    # plot misclassification
    arr=[]
    for i in range(test_y_raw.shape[0]):
        if test_y_raw[i]!=pred_y[i]:
            arr.append(i)
    i=arr[np.random.randint(len(arr))]
    plt.imshow(test_x_raw[i])
    plt.title(('True: ', test_y_raw[i], 'Predict: ', pred_y[i]))
    plt.show()