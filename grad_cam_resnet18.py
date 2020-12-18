import argparse
import cv2
import os
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import pretrainedmodels
from cifar10_resnet18_MyRelu import ResNet,ResidualBlock
from small_cnn import CNN
# from nobn_cnn import N_BN_CNN


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

def show_cam_on_image(img, mask, name, layer):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("4CNN2/epoch50/{}/{}_cam_{}.jpg".format(name, name, layer), np.uint8(255 * cam))

def show_cam_on_image_1000(img, mask, path,imagename, layer):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path+"/{}_cam_{}.jpg".format(imagename, layer), np.uint8(255 * cam))

def show_cam_on_image1(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("testsenet_cam_l2.jpg", np.uint8(255 * cam))


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

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
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        print('种类：',target,index)
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # cam = np.where(cam < 0.002, 0, cam)
        # 数值归一化
        cam -= np.min(cam)
        if np.max(cam) != 0:
            cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (32, 32))
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




def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def read_directory(directory_name):
    imagesdata = []
    root_path  = []
    for filename in os.listdir(directory_name):
        picturename = os.path.join(basicdata,filename)
        print(picturename)
        root_path.append(picturename)
        List = []
        for picturs in os.listdir(picturename):
            pictures = os.path.join(picturename, picturs)
            List.append(pictures)
        imagesdata.append(List)
    print(len(imagesdata),imagesdata)
    print(len(root_path), root_path)
    return root_path,imagesdata




if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # args = get_args()
    cifar10 = ['airplane', 'car', 'bird',  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # image_path = './examples/ILSVRC2012_val_00029365.jpeg'
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    # model = N_BN_CNN()
    # model.load_state_dict(torch.load('N_BN_CNN.ckpt'))
    # model = ResNet(ResidualBlock, [2, 2, 2, 2])
    # model.load_state_dict(torch.load('resnet18_test_epoch50.ckpt'))
    model = CNN()
    model.load_state_dict(torch.load('4CNN_50.ckpt'))
    # model = models.resnet50(pretrained=True)
    # model_name = 'senet154'  # could be fbresnet152 or inceptionresnetv2
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    layer_4 = ['conv1', 'conv2', 'conv3','conv4']
    layer = ['conv','layer1','layer1.0','layer1.0.conv1','layer1.0.conv2','layer1.1','layer1.1.conv1','layer1.1.conv2','layer2',
             'layer2.0','layer2.0.conv1','layer2.0.conv2','layer2.0.downsample.0','layer2.0.downsample.1','layer2.1','layer2.1.conv1',
             'layer2.1.conv2','layer3','layer3.0','layer3.0.conv1','layer3.0.conv2','layer3.0.downsample','layer3.0.downsample.0','layer3.0.downsample.1',
             'layer3.1','layer3.1.conv1','layer3.1.conv2','layer4','layer4.0','layer4.0.conv1','layer4.0.conv2','layer4.0.downsample',
             'layer4.0.downsample.0','layer4.0.downsample.1','layer4.1','layer4.1.conv1','layer4.1.conv2']
    # grad_cam = GradCAM(net=model, layer_name='layer2')
    #
    # img = cv2.imread(image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    # # input = torch.from_numpy(img)
    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = None
    # mask = grad_cam(input, target_index)
    #
    # show_cam_on_image1(img, mask)
    # #
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    # print(model._modules.items())
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    #cv2.imwrite('small_cnn_test_100/cat_gb1_4.jpg', gb)
    #cv2.imwrite('small_cnn_test_100/cat_cam_gb1_4.jpg', cam_gb)
    #cifar10grad-cam实验
    for i in range(len(cifar10)):
        for j in range(len(layer_4)):
            image_path = './examples/{}.jpg'.format(cifar10[i])
            grad_cam = GradCAM(net=model, layer_name=layer_4[j])
            target_index = i
            img = cv2.imread(image_path, 1)
            img = np.float32(cv2.resize(img, (32, 32))) / 255
            input = preprocess_image(img)
            mask = grad_cam(input, target_index)
            show_cam_on_image(img, mask, cifar10[i], layer_4[j])

    #imagine1000 grad_cam实验测试：
    # basicdata = 'F:\ILSVRC2012\small_val'
    # root_path,images = read_directory(basicdata)
    # layer_image1000 = 'layer4'
    # for i in range(len(images)):
    #     for j in range(len(images[i])):
    #         image_path = images[i][j]
    #         grad_cam = GradCAM(net=model, layer_name=layer_image1000)
    #         target_index = None
    #         img = cv2.imread(image_path, 1)
    #         image_name=images[i][j].split('\\')[4].split('.')[0]
    #         img = np.float32(cv2.resize(img, (224, 224))) / 255
    #         input = preprocess_image(img)
    #         mask = grad_cam(input, target_index)
    #         savepath = 'F:\ILSVRC2012\\senet50_grad_cam\{}'.format(root_path[i].split('\\')[3])
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)
    #         show_cam_on_image_1000(img, mask,savepath,image_name,layer_image1000)