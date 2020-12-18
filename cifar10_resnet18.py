import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
import cv2
import math
# from nobn_cnn import N_BN_CNN
from small_cnn import CNN
# Device configuration


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input



device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
num_epochs = 100
batch_size = 64
learning_rate = 0.001
relu = MyReLU.apply
# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                             train=True,
                                             transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
print(len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def conv3x3(in_channels, out_channels, strides=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=strides, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self,block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, strides = stride),
                nn.BatchNorm2d(out_channels))

        layers=[]
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out




if __name__ == '__main__':

    model = ResNet(ResidualBlock, [2,2,2,2]).to(device)
    model.load_state_dict(torch.load('resnet18_test.ckpt'))
    summary(model, (3, 32, 32))
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


    # def update_lr(optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #
    # total_step = len(train_loader)
    # curr_lr = learning_rate
    # model.train()
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #
    #         # forward pass
    #         output = model(images)
    #         loss = criterion(output,labels)
    #
    #         #Backword pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if (i+1)%100 == 0:
    #             print("Epoch:[{}/{}],Step:[{}/{}], Loss:{:.4f}"
    #             .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    #
    #     if (epoch+1)%20 == 0:
    #         curr_lr /= 3
    #         update_lr(optimizer,curr_lr)
    # total_step = len(test_loader)
    # print(total_step)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


    # Save the model checkpoint
    # torch.save(model.state_dict(), 'resnet18_test_epoch0_test.ckpt')
















