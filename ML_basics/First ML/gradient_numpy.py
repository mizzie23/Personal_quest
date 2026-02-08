import numpy as np
# manual linear regression
#f = w•x let w = 2
X = np.array([1,2,3], dtype = np.float32) # input 
Y = np.array([2,4,6], dtype = np.float32) #y_actual

w = 0.0

#prediction model
def forward(x):
    return w*x

#loss ( mean square error)

def loss(y, y_prediction):
    return ((y_prediction - y)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N* 2*x*(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted - y).mean()



print(f'prediction before training: f(5) = {forward(5):.3f}')




# training
learning_rate = 0.01
n_iters = 40

for epoch in range(n_iters):
    #predction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient
    dw = gradient(X,Y, y_pred)

    #update weights
    w -= learning_rate*dw
 

    if epoch%2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print (f'prediction after training: f(5) = {forward(5):.3f}')






#now to do the same with pytorch is quiten easy. will do it soon