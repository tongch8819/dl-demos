import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import math

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

def _extract_tensors(dset, num=None):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Inputs:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: [Optional]. If provided, the number of samples to keep

    Returns:
    - x: float32 tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=torch.float32).permute(0,3,1,2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError('Invalid value num=%d; must be in the range [0, %d]'
                             % (num, x.shape[0]))
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y

def cifar10(num_train=None, num_test=None):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: [Optional] How many samples to keep from the training set.
      If not provided, then keep the entire training set.
    - num_test: [Optional] How many samples to keep from the test set.
      If not provided, then keep the entire test set.

    Returns:
    - x_train: float32 tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: float32 tensor of shape (num_test, 3, 32, 32)
    _ y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    download = not os.path.isdir('cifar-10-batches-py')
    dset_train = CIFAR10(root='.', download=download, train=True)
    dset_test = CIFAR10(root='.', train=False)
    x_train, y_train = _extract_tensors(dset_train, num_train)
    x_test, y_test = _extract_tensors(dset_train, num_test)
    
    return x_train, y_train, x_test, y_test


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]
    
    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to('cpu', torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot.

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch, ] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names
      e.g.) ['plane', 'car']

    Outputs:
    - An grid-image that visualize samples_per_class number of samples per class
    """
    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(-4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha='right')
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])
    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)

def get_CIFAR10_data(
    validation_ratio=0.05,
    cuda=False,
    reshape_to_2d=False,
    visualize=False
):
    X_train, y_train, X_test, y_test = cifar10(
        num_train=None,
        num_test=10000,
    )

    # Load every data on cuda
    if cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    # 0. Visualize some examples from the dataset
    class_names = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    if visualize:
        img = visualize_dataset(X_train, y_train, 12, class_names)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    # 1. Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=0, keepdim=True).mean(
        dim=2, keepdim=True).mean(dim=3, keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # 2. Reshape the image data into rows
    if reshape_to_2d:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Take the validation set from the training set
    # Note: It should not be taken from the test set
    num_training = int( X_train.shape[0] * (1.0 - validation_ratio) )
    num_validation = X_train.shape[0] - num_training

    # return the dataset
    data_dict = {}
    data_dict['X_val'] = X_train[num_training:num_training + num_validation]
    data_dict['y_val'] = y_train[num_training:num_training + num_validation]
    data_dict['X_train'] = X_train[0:num_training]
    data_dict['y_train'] = y_train[0:num_training]
    data_dict['X_test'] = X_test
    data_dict['y_test'] = y_test

    return data_dict



class VGGNetEleven(nn.Module):
    def __init__(self):
        super(VGGNetEleven, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.ReLU(x)

        x = F.max_pool2d(x)

        x = self.conv2(x)
        x = F.ReLU(x)

        x = F.max_pool2d(x)

        x = self.conv3(x)
        x = F.ReLU(x)
        x = self.conv3(x)
        x = F.ReLU(x)

        x = F.max_pool2d(x)

        x = self.conv4(x)
        x = F.ReLU(x)
        x = self.conv4(x)
        x = F.ReLU(x)

        x = F.max_pool2d(x)

        x = self.conv4(x)
        x = F.ReLU(x)
        x = self.conv4(x)
        x = F.ReLU(x)
        
        x = F.max_pool2d(x)

        x = self.fc1(x)
        x = F.ReLU(x)
        x = self.fc1(x)
        x = F.ReLU(x)
        x = self.fc2(x)
        output = F.softmax(x)
        return output

class VGG(nn.Module):
    def __init__(self, features):
        """
        Inputs:
        - features: nn.Sequential object represents variation of vgg-X.
        """
        super(VGG, self).__init__()
        flat_channels = 512
        num_classes = 10
        self.features = features 
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flat_channels, flat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(flat_channels, flat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(flat_channels, num_classes),
            nn.Softmax(dim=1)
        )
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2/n))
                m.bias.data.zero_()

    def forward(self, x):
        """
        Don't forget the flatten operation between Convolution stage and FC stage.

        Input:
        - x: tensor image
        """
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        output = self.classifier(x)
        return output 

def build_feature_layers(config, batch_norm=False):
    """
    Build the feature layers according to the vgg_configs.

    Inputs:
    - vgg_configs: list of keywords, int or char
    - batch_norm: [Optional] switch of batch normalization

    Outputs:
    - a Sequential object with all the layers inside
    """
    layers = []
    in_channels = 3
    for token in config:
        if token == 'M':
            layers.append( nn.MaxPool2d(kernel_size=2, stride=2) )
        else:
            out_channels = token
            conv2d = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            assert type(token) == int, "Int token type ERROR!"
            layers.append( conv2d )
            if batch_norm:
                layers.append( nn.BatchNorm2d(out_channels) )
            layers.append( nn.ReLU(inplace=True) )
            in_channels = out_channels
    return nn.Sequential( *layers )


vgg_configs = {
    "A" : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "B" : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "C" : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "D" : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11():
    return VGG(build_feature_layers(vgg_configs['A']))

def vgg11_bn():
    return VGG(build_feature_layers(vgg_configs['A'], True))

def vgg13():
    return VGG(build_feature_layers(vgg_configs['B']))

def vgg13_bn():
    return VGG(build_feature_layers(vgg_configs['B'], True))

def vgg16():
    return VGG(build_feature_layers(vgg_configs['C']))

def vgg16_bn():
    return VGG(build_feature_layers(vgg_configs['C'], True))

def vgg19():
    return VGG(build_feature_layers(vgg_configs['D']))

def vgg19_bn():
    return VGG(build_feature_layers(vgg_configs['D'], True))


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    # switch model to train mode
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

def validate(val_loader, model, criterion, device):
    # switch model to evaluation mode
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            val_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    acc = correct / len(val_loader.dataset)
    print("\nAverage loss of validation set: {:.4f}; Accuracy: {}/{} ({:.0%})\n".format(
        val_loss,
        correct,
        len(val_loader.dataset),
        acc
    ))

    return acc

def main():
    data_dict = get_CIFAR10_data(visualize=False)
    
    # print('Train data shape: ', data_dict['X_train'].shape)
    # print('Train labels shape: ', data_dict['y_train'].shape)
    # print('Validation data shape: ', data_dict['X_val'].shape)
    # print('Validation labels shape: ', data_dict['y_val'].shape)
    # print('Test data shape: ', data_dict['X_test'].shape)
    # print('Test labels shape: ', data_dict['y_test'].shape)

    args = {
        'batch_size' : 128,
        'lr' : 0.05,
        'log_interval' : 10,
        'epochs' : 500,
        'cuda' : True,
        'save_model' : True,
    }
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    use_cuda = args['cuda'] and torch.cuda.is_available()
    device = torch.device('cpu' if not use_cuda else 'cuda')
    print("Device:", device)
    
    train_loader_args = {
        'batch_size' : args['batch_size'],
    }
    val_loader_args = {
        'batch_size' : args['batch_size'],
    }
    if use_cuda:
        cuda_kwargs = {
            'num_workers' : 2,
            'pin_memory' : True,
            'shuffle' : True,
        }
        train_loader_args.update(cuda_kwargs)
        val_loader_args.update(cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        **train_loader_args
    )
    
    val_loader = torch.utils.data.DataLoader(
        CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        **val_loader_args
    )
    
    # model = vgg11().to(device)
    model = vgg13().to(device)
    # model = vgg16().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args['lr'])
    # optimizer = torch.optim.Adagrad(model.parameters(), args['lr'])

    best_model = None
    best_acc = 0.0
    for epoch in range(1, args['epochs'] + 1):
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        acc = validate(val_loader, model, criterion, device)

        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)

    if args['save_model']:
        torch.save(best_model.state_dict(), 'model/cifar10_vgg13.pt')
        print("Model Saved Success.")
    
    print("Well done!!!")

if __name__ == "__main__":
    main()