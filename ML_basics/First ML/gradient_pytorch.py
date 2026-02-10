#1) Design model (input, output size, forward pass)
#2) construct loss and optimizer
#3) training loop
#    - forward pass : compute prediction
#    - backward pass : calculate gradient
#    - update weights

import torch
import torch.nn as nn
# pytorch autograder

X = torch.tensor([1,2,3], dtype = torch.float32) # input 
Y = torch.tensor([2,4,6], dtype = torch.float32) #y_actual

w = torch.tensor(0.0, requires_grad=True)

#prediction model(manual)

def forward(x): 
    return w*x

#loss ( mean square error)

# def loss(y, y_prediction):
    # return ((y_prediction - y)**2).mean()

print(f'prediction before training: f(5) = {forward(5):.3f}')




# training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss() #no need to write the formula manually
optimizer = torch.optim.SGD([w] , lr = learning_rate)

for epoch in range(n_iters):
    #predction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    l.backward()

    #update weights
                         # with torch.no_grad():
                         #     w -= learning_rate * w.grad
    optimizer.step()
    
    #zero_gradients
    # w.grad.zero_()
    optimizer.zero_grad()
 

    if epoch%10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print (f'prediction after training: f(5) = {forward(5):.3f}')






#Prediction of Model
# Gradients Computation 
# loss Computation
# Parameter updates
