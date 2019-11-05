import os
import sys
import torch
from torch.autograd import Variable
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import init
#from visdom import Visdom


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class DenseNormReLU(nn.Module):
    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(DenseNormReLU, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features = in_feats, out_features = out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class VGGM(nn.Module):
    def __init__(self, num_class,dims=256):
        super(VGGM,self).__init__()
        self.conv1 = nn.Conv2d(3,96,kernel_size=7,stride=2)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,stride=2,padding=1)
        self.conv3 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.max_pool1 = nn.MaxPool2d(3,2)
        self.max_pool2 = nn.MaxPool2d(3,2)
        self.max_pool3 = nn.MaxPool2d(3,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*25,4096)
        self.fc2 = nn.Linear(4096,1024)
        self.fc3 = nn.Linear(1024,num_class)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc_head = DenseNormReLU(in_feats = 1024, out_feats = 512)
        #self.embed = nn.Linear(in_features = 512, out_features = dims)

    def forward(self,x):
        x = self.conv1(x)
        conv1 = F.relu(x)
        x = self.bn1(conv1)
        pool1 = self.max_pool1(x)
        x = self.conv2(pool1)
        conv2 = F.relu(x)
        x = self.bn2(conv2)
        pool2 = self.max_pool2(x)
        x = self.conv3(pool2)
        conv3 = F.relu(x)
        x = self.conv4(conv3)
        conv4 = F.relu(x)
        x = self.conv5(conv4)
        conv5 = F.relu(x)
        pool5 = self.max_pool3(conv5)
        x = pool5.view(-1,512*25)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        n_x = F.normalize(x, p=2, dim=1)
        #n_x = n_x * 10
        #n_x = self.fc_head(x)
        #n_x = self.embed(n_x)
        # y = self.fc3(x)
        return conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, n_x

def vggm(num_class):
    model=VGGM(num_class)
    model.apply(weights_init_normal)
    return model








