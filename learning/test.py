'''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


m = nn.Linear(2, 3)
input = torch.randn(6, 2)
#print(input)
#print()
output = m(input)
#print(output)
#print(output.size())


weight = Parameter(torch.Tensor(2, 3))
print(weight.size(1))
'''


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

x = np.array([-2.2, -.8])
y = np.array([0.0, 1.0])

#x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
#y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

logr = LogisticRegression(solver='lbfgs')
logr.fit(x.reshape(-1, 1), y)
print(logr.predict_proba(x.reshape(-1, 1)))

y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()
loss = log_loss(y, y_pred)

print('x = {}'.format(x))
print('y = {}'.format(y))
print('p(y) = {}'.format(np.round(y_pred, 2)))
print(y_pred)
print(loss)
print('Log Loss / Cross Entropy = {:.4f}'.format(loss))