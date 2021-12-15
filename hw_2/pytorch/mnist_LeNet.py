import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        # output = F.log_softmax(x, dim=1)
        output = x
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    loss_hist = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensor onto device
        data, target = data.to(device), target.to(device)
        # clear optimizer gradient
        optimizer.zero_grad()
        # make prediction
        output = model(data)
        # compute loss
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        # trigger backpropogation
        loss.backward()
        # update optimizer
        optimizer.step()
        # print log info
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
        loss_hist.append(loss.item())
    return loss_hist
        

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # in no gradient mode
    with torch.no_grad():
        for data, target in test_loader:
            # move tensor onto device
            data, target = data.to(device), target.to(device)
            # make prediction
            output = model(data)
            # update total loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    len_test_set = len(test_loader.dataset)
    test_loss /= len_test_set
    acc = correct / len_test_set
    print('\nTest set: Average loss: {:.4f}, Accuracy, {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * acc
    ))
    return test_loss, acc

def plot_loss_hist(loss_hist, args):
    x = range(len(loss_hist))
    plt.plot(x, loss_hist)
    plt.title(args['title'])
    plt.xlabel('batch number')
    plt.ylabel('cross entropy loss')
    plt.savefig(args['path'])

def main():
    args = {
        'batch_size' : 128,
        'test_batch_size' : 1000,
        'epochs' : 5, 
        'lr' : 0.1,
        'gamma' : 0.7,
        'cuda' : True,
        'dry_run' : False,
        'seed' : 1,
        'log_interval' : 10,
        'save_model' : True
    }
    
    # specify the tensor device
    use_cuda = args['cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)
    
    torch.manual_seed(args['seed'])
    
    # preprare data
    train_kwargs = { 'batch_size' : args['batch_size'] }
    test_kwargs = { 'batch_size' : args['test_batch_size'] }
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory' : True,
            'shuffle' : True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # define a transform for each PIL image
    # make a composition of two transform operation
    transform = transforms.Compose([
        transforms.Pad((2,2)),
        transforms.ToTensor(),
        # transforms.Normalize((0, ), (255, )) # from [0, 255] to [0, 1]
    ])
    
    dataset1 = datasets.MNIST('./data', 
        train=True, 
        download=True,
        transform=transform
    )
    dataset2 = datasets.MNIST('./data', 
        train=False, 
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    model = LeNet()
    # for batch in train_loader:
    #     images, targets = batch
    #     print(targets.shape)
    #     print(targets)
    #     output = model(images[:16])
    #     print(output.shape)
    #     print(output)
    #     print("Loss:", F.cross_entropy(output, targets))
    #     b = F.log_softmax(output, dim=1)
    #     print("Loss:", F.nll_loss(b, targets))
    #     break
    
    
    # exit()
    
    model = LeNet().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
    # optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    best_model = None
    best_acc = 0.0
    epoch_table = []
    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
    for epoch in range(1, args['epochs'] + 1):
        loss_hist = train(args, model, device, train_loader, optimizer, epoch)
        plot_hist_args = {
            'title' : "Epoch " + str(epoch),
            'path' : "plots/epoch-{}.jpg".format(epoch)
        }
        plot_loss_hist(loss_hist, plot_hist_args)
        test_loss, acc = test(model, device, test_loader)
        a_lr = scheduler.get_last_lr()[0]
        epoch_table.append([a_lr, test_loss, acc])
    
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)
    
        scheduler.step()
    
    df = pd.DataFrame(epoch_table, index=range(1, args['epochs']+1), columns=['LR', 'Test Loss', 'ACC'])
    df.to_csv("csv/lenet_mnist_pytorch.csv")
    
    if args['save_model']:
        torch.save(best_model.state_dict(), "model/mnist_lenet.pt")
        print("Model Saved Success")
    
    print("Well done!!")


if __name__ == "__main__":
    main()