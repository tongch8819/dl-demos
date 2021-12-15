from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from copy import deepcopy

# Extra Point
# 1. ReLU activation
# 2. dropout layer
# 3. Adagrad

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # print(target.shape)
        # exit()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
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
            if args['dry_run']:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy, {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * acc
    ))
    return acc

def main():
    args = {
        'batch_size' : 128,
        'test_batch_size' : 1000,
        'epochs' : 14,
        'lr' : 0.01,
        'gamma' : 0.7,
        'no_cuda' : False,
        'dry_run' : False,
        'seed' : 1,
        'log_interval' : 10,
        'save_model' : True
    }
    
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    
    torch.manual_seed(args['seed'])
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)
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
        transforms.ToTensor(),
        # transforms.Normalize((0.1307, ), (0.3081, ))
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
    
    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
    
    best_model = None
    best_acc = 0.0
    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)
        scheduler.step()
    
    if args['save_model']:
        torch.save(best_model.state_dict(), "model/mnist_fc.pt")
        print("Model Saved Success")
    
    print("Well done!!")

if __name__ == "__main__":
    main()