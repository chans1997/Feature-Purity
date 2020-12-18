import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import math
import pretrainedmodels
from torchsummary import summary

batch_size = 1

data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])


# train_dataset =torchvision.datasets.ImageFolder(root='ILSVRC2012/train',transform=data_transform)
# train_dataset_loader =DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=4)

test_dataset = torchvision.datasets.ImageFolder(root='F:/ILSVRC2012/small_val',transform=data_transform)
test_dataset_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False,num_workers=4)


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i]-means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i]/stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    print(input.shape)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("single_cam_test.jpg", np.uint8(255 * cam))


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, use_cuda=True):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.cuda = use_cuda
        if self.cuda:
            self.net = net.cuda()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs.cuda())  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()


        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        # cam = self.feature[0].cpu().data.numpy()
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # cam = np.mean(cam, axis=(1, 2))
        # print(np.min(cam))
        # print(np.max(cam))


        # # 数值归一化
        for i in range(cam.shape[0]):
            # cam[i, :, :] -= np.min(cam[i, :, :])
            if np.max(cam[i, :, :]) != 0:
                cam[i, :, :] /= np.max(cam[i, :, :])
        print(np.min(cam))
        print(np.max(cam))
        cam = np.mean(cam, axis=(1, 2))
        # cam -= np.min(cam)
        # cam /= np.max(cam)
        # # resize to 224*224
        # cam = cv2.resize(cam, (224, 224))
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        pasitive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, pasitive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)


    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def compute_comentropy(mean,thre):
    bin = np.zeros(shape=(1,mean.shape[1]))

    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if mean[i][j] > thre:
                bin[0][j] = bin[0][j]+1
    sum = bin.sum()
    print('sum:', sum)
    bin_normal = bin/bin.sum(axis=1)
    count = bin[bin>0].shape[0]
    for i in range(bin_normal.shape[1]):
        if bin_normal[0][i] > 0:
           bin_normal[0][i] = bin_normal[0][i]*np.log2(bin_normal[0][i])
    fin = -bin_normal.sum()
    return bin, bin_normal, fin,count

def filter_entropy_compute(mean):
    total = []
    bin = np.zeros(shape=(1,5))
    mean = np.transpose(mean)
    print('特征层',mean.shape)
    for i in range(mean.shape[0]):
        bin[0, 0] = np.array(np.where(mean[i, :] <= 0.1)).shape[1]
        bin[0, 1] = np.array(np.where(mean[i, :] <= 0.2)).shape[1]-np.array(np.where(mean[i, :] <= 0.1)).shape[1]
        bin[0, 2] = np.array(np.where(mean[i, :] <= 0.3)).shape[1]-np.array(np.where(mean[i, :] <= 0.2)).shape[1]
        bin[0, 3] = np.array(np.where(mean[i, :] <= 0.4)).shape[1]-np.array(np.where(mean[i, :] <= 0.3)).shape[1]
        bin[0, 4] = np.array(np.where(mean[i, :] > 0.4)).shape[1]
        print(bin)
        bin /= bin.sum(axis=1)
        print(bin)
        for i in range(bin.shape[1]):
            if bin[0][i] > 0:
                bin[0][i] = bin[0][i] * np.log2(bin[0][i])
        total.append(-bin.sum())
    print(np.array(total).shape)
    max_entropy = (-len(total)*(np.log2(0.2)))
    print('最大值',max_entropy)
    print('sum:',sum(total))
    return sum(total)/max_entropy



def softmax_compute_entropy(feature, thre):
    # if 'fc' in feat:
    #     mean = feature
    # else:
    mean = np.mean(feature, axis=(2, 3))
    print(mean, mean.shape)
    # mean = mean-np.max()
    #mean = np.maximum(mean, 0)
    mean -= np.max(mean, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    mean = np.exp(mean) / np.sum(np.exp(mean), axis=1, keepdims=True)
    #mean = np.maximum(mean, 0)
    print(mean, mean.shape)
    bin = np.zeros(shape=(1, mean.shape[1]))
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if mean[i][j] > thre:
                bin[0][j] = bin[0][j] + 1
    sum = bin.sum()
    print('sum:', sum)
    bin_normal = bin / bin.sum()
    for i in range(bin_normal.shape[1]):
        if bin_normal[0][i] > 0:
            bin_normal[0][i] = bin_normal[0][i] * np.log2(bin_normal[0][i])
    # inf = np.log2(bin_normal)
    # fin =np.multiply(bin_normal,inf)
    fin = -bin_normal.sum()
    return bin, bin_normal, fin

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)



if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # args = get_args()
    # image_path = './examples/batch_1_num_809.jpg'
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.densenet161(pretrained=True)
    # model = N_BN_CNN()
    # model.load_state_dict(torch.load('N_BN_CNN.ckpt'))
    # model = ResNet(ResidualBlock, [2, 2, 2, 2])
    # model.load_state_dict(torch.load('resnet18_test_epoch50.ckpt'))
    # model_name = 'senet154'  # could be fbresnet152 or inceptionresnetv2
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # summary(model,input_size=(3,224,224))
    grad_cam = GradCAM(net=model, layer_name='features.denseblock3')

    # img = cv2.imread(image_path, 1)
    # img = np.float32(cv2.resize(img, (32, 32))) / 255
    # input = preprocess_image(img)
    # input = torch.from_numpy(img)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask_sum = []
    for i, (images, labels) in enumerate(test_dataset_loader):
        images = images.cuda()
        labels = labels.cuda()
        images.requires_grad_(True)
        mask = grad_cam(images, target_index)
        mask_sum.append(mask)
    print((np.array(mask_sum)).shape)
    mask_sum = np.array(mask_sum)
    # threh = [0.001, 0.002, 0.003, 0.004, 0.005]#,0.007, 0.01, 0.02, 0.05, 0.1]
    threh = [0.1, 0.2, 0.3, 0.4, 0.5]
    sum_total = []
    for i in range(len(threh)):
        bin, bin_normal, fin, count = compute_comentropy(mask_sum, threh[i])
        print(bin, bin.shape)
        print(bin_normal, bin_normal.shape)
        print("filter激活数量：",count)
        print('信息熵:', fin)
        m = -math.log2(1 / bin.shape[1])
        print('最大信息熵：', m)
        print('信息熵分数：%.2f' % (fin / m))
        sum_total.append(round(fin/m, 3))
    print(sum_total)
    # tensor_entropy = filter_entropy_compute(mask_sum)
    # print('特征层分数',tensor_entropy)
    # s = np.zeros(shape=(0, total.shape[1], total.shape[2], total.shape[3]))
    # for idx in range(len(mask_sum) - 1):
    #     total = np.concatenate((np.array(mask_sum[idx]), total), axis=0)
    #
    # print(total.shape)
    # show_cam_on_image(img, mask)
    #
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    # print(model._modules.items())
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    #cv2.imwrite('small_cnn_test_100/cat_gb1_4.jpg', gb)
    #cv2.imwrite('small_cnn_test_100/cat_cam_gb1_4.jpg', cam_gb)



