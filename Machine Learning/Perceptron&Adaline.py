class Perceptron(object):
    def __init__(self, lr=0.1, epoch=10):
        self.lr = lr
        self.epoch = epoch

    def fit(self, x, y):
        self.n_classes = len(np.unique(test_y))
        self.w = np.zeros((x.shape[1]+1,self.n_classes))
        for _ in range(self.epoch):
            for xi, yi in zip(x,y):
                dw = self.lr*(yi-self.predict(xi))
                self.w[1:,yi] += dw*xi
                self.w[0,yi]  += dw

    def predict(self, v):
        tmp = np.dot(v, self.w[1:,:])+self.w[0,:]
        try:
            return np.argmax(tmp, axis=1)
        except:
            return np.argmax(tmp)
        
class Adaline(object):
    def __init__(self,lr=0.1, epoch=10):
        self.lr = lr
        self.epoch = epoch

    def fit(self, x, y):
        self.n_classes = len(np.unique(test_y))
        self.w = np.zeros((x.shape[1]+1,self.n_classes))
        for xi, yi in zip(x, y):
            for _ in range(self.epoch):
                dw = self.lr*(yi-self.net_input(xi,yi))
                self.w[1:,yi] += dw*xi
                self.w[0,yi] += dw

    def net_input(self, x, y):
        return np.dot(x, self.w[1:,y]+self.w[0,y])
   
    def predict(self, v):
        tmp = np.dot(v, self.w[1:,:])+self.w[0,:]
        print(tmp)
        try:
            return np.argmax(tmp, axis=1)
        except:
            return np.argmax(tmp)
