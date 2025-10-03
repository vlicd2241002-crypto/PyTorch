import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Lectura de la DB MNIST 

import pandas as pd
train = pd.read_csv("train.csv")
train_labels = train['label'].values
train = train.drop("label",axis=1).values.reshape(len(train),1,28,28)

#Empaquetado en Tensores
X = torch.Tensor(train.astype(float))
y = torch.Tensor(train_labels).long()

#Definición del modelo
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 98)
        self.fc4 = nn.Linear(98, 10)

        #Con el fin de regularizar la red neuronal y prevener overfitting
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        output = F.log_softmax(self.fc4(x), dim=1)
        return output


#Parametrización del modelo
model = MNISTClassifier()
loss_function = nn.NLLLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

#Entrenamiento de la red neuronal
for epoch in range(50):
    images = Variable(X)
    labels = Variable(y)
    opt.zero_grad()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    #loss = F.nll_loss(outputs, labels)
    loss.backward()
    opt.step()
    print ('Epoch [%d/%d] Loss: %.4f' %(epoch+1, 50, loss.data.item()))