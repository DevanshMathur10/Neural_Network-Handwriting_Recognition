import numpy as np
import mnist_loader
import random
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self,sizes):
        self.sizes=sizes
        self.num_layers=len(self.sizes)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedForward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a

    def Stochastic_Gradient_Descent(self,training_data,epochs,mini_batch_size,eta,test_data=None):

        if test_data:
            n_test=len(test_data)

        n=len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            if test_data:
                print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete.")
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        
    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        activation=x
        activations=[x]
        zs=[]

        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])

        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10])
net.Stochastic_Gradient_Descent(training_data, 30, 10, 7)

x=int(input("Enter a number between 1-9999 : "))

while x!=0:
    image=test_data[x][0].reshape(28,28)
    plt.imshow(image, cmap='gray')
    num=np.argmax(net.feedForward(test_data[x][0]))
    plt.title(f"My neural network predicts {num} !")
    plt.show()
    x=int(input("Enter a number between 1-9999 : "))

print(f"Accuracy : {net.evaluate(test_data)/100} %")

""" 
LR=3 | Accuracy=94.80%
LR=4 | Accuracy=94.77%
LR=5 | Accuracy=95.45%
LR=6 | Accuracy=95.02%
LR=7 | Accuracy=95.56%
LR=8 | Accuracy=95.10%
LR=10 | Accuracy=94.59%
LR=15 | Accuracy=94.42%
"""