class LVQ1(object):
    def __init__(self, lr=0.5, epoch=10):
        self.lr = lr
        self.epoch=epoch
    
    # calculate the distance between two vectors
    def dist(self, a, b):
        return np.sqrt(sum((a-b)**2))
    
    def fit(self, x, y, n=3):
        self.w=np.random.rand(n,x.shape[1])
        print('Initial weight is:\n', self.w)
        # repeeat
        for t in range(self.epoch):
            # adjust the learning rate
            lr=self.lr*(1-t/self.epoch)
            for (xi,yi) in zip(x,y):
                d=[]
                for i in range(self.w.shape[0]):
                    d.append(self.dist(xi,self.w[i]))
                # find node j*
                j=np.argmin(d)
                # update the weight w[j]
                if j==yi:
                    dw=lr*(xi-self.w[j])
                else:
                    dw=-lr*(xi-self.w[j])
                self.w[j]+=dw
        return self
    
    def predict(self, v):
        pred=[]
        for i in range(v.shape[0]):
            d=[]
            for j in range(self.w.shape[0]):
                d.append(self.dist(v[i],self.w[j]))
            pred.append(np.argmin(d))
        return pred
