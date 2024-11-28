"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define your model architecture

        #target.py architecture
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,stride=2,padding=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=8,kernel_size=5,stride=2,padding=2)
        self.fc_1 = nn.Linear(in_features=32,out_features=2)

        #source.py architecture
        #self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2)
        #self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,stride=2,padding=2)
        #self.conv3 = nn.Conv2d(in_channels=64,out_channels=8,kernel_size=5,stride=2,padding=2)
        #self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        #self.fc1 = nn.Linear(in_features=32,out_features=8)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        ## TODO: initialize the parameters for your network
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        #target.py
        self.fc_1.weight.data.normal_(mean=0,std=1/sqrt(32))
        self.fc_1.bias.data.fill_(0.0)

        #source.py
        #self.fc1.weight.data.normal_(mean=0,std=1/sqrt(32))
        #self.fc1.bias.data.fill_(0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape

        ## TODO: implement forward pass for your network

        layer1 = F.relu(self.conv1(x))
        pool1 = self.pool(layer1)

        

        layer2 = F.relu(self.conv2(pool1))
        pool2 = self.pool(layer2)


        layer3 = F.relu(self.conv3(pool2))
        layer3_flat = torch.flatten(layer3,1)

        #target.py
        out = self.fc_1(layer3_flat)
        #source.py
        #out = self.fc1(layer3_flat)
        return out
