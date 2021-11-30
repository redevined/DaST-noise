import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg import VGG

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
trainset = torchvision.datasets.CIFAR10(root='dataset/', train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                            # transforms.Pad(2, padding_mode="symmetric"),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize,
                                    ]))
testset = torchvision.datasets.CIFAR10(root='dataset/', train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                    ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = False, num_workers = 2)

model = VGG('VGG16').cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .5, patience = 10)

def train_epoch(train = True) :
    if train :
        model.train()
        loader = trainloader
    else :
        model.eval()
        loader = testloader
    total = 0
    total_loss = 0
    correct = 0
    
    for inputs, targets in loader :
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if train :
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += targets.size(0)
        total_loss += loss.item()
        pred = outputs.argmax(dim = 1)
        correct += (pred == targets).sum().item()
    
    return total_loss / total, correct / total

epoch = 1
best_acc = 0
while optimizer.param_groups[0]['lr'] > 1e-4 :
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = train_epoch(False)
    scheduler.step(test_acc)

    print(f'Epoch {epoch}: train_loss = {train_loss}, train_acc = {train_acc}, test_loss = {test_loss}, test_acc = {test_acc}')
    if test_acc > best_acc :
        best_acc = test_acc
        torch.save(model.state_dict(), 'pretrained/better_vgg16_cifar10.pth')
    epoch += 1