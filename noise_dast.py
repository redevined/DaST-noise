from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack
# from sklearn.externals import joblib
import joblib
# from utils import load_data
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from net import Net_s, Net_m, Net_l
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34
cudnn.benchmark = True
# workbook = xlwt.Workbook(encoding = 'utf-8')
# worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

# sys.stdout = Logger('imitation_network_model_noise.log', sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='azure')
parser.add_argument('--niter', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='tradeoff factor between matching label (beta=0) vs. output distribution (beta=1)')
parser.add_argument('--data_dist', type=str, default='normal', help='distribution to sample input data from')
parser.add_argument('--save_folder', type=str, default='saved_model_noise', help='alpha')

opt = parser.parse_args()
print(opt)

# if not os.path.exists(opt.save_folder) :
#     os.makedirs(opt.save_folder)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'azure':
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
    netD = Net_l().cuda()
    netD = nn.DataParallel(netD)

    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')

    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

elif opt.dataset == 'mnist':
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
    netD = Net_l().cuda()
    netD = nn.DataParallel(netD)

    original_net = Net_m().cuda()
    state_dict = torch.load(
        'pretrained/net_m.pth')
    original_net.load_state_dict(state_dict)
    original_net = nn.DataParallel(original_net)
    original_net.eval()

    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

elif opt.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testset = torchvision.datasets.CIFAR10(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                normalize,
                                        ]))
    netD = ResNet50().cuda()
    netD = nn.DataParallel(netD)

    original_net = VGG('VGG16').cuda()
    state_dict = torch.load(
        'pretrained/better_vgg16_cifar10.pth')
    original_net.load_state_dict(state_dict, strict = False)
    original_net = nn.DataParallel(original_net)
    original_net.eval()

    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

data_list = [i for i in range(6000, 8000)] # fast validation
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)
# nc=1

device = torch.device("cuda:0" if opt.cuda else "cpu")
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data)
    output = torch.from_numpy(output).cuda().long()
    return output

def cal_azure_proba(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict_proba(data)
    output = torch.from_numpy(output).cuda().float()
    return output

# Sample random data from distribution
def sample_random_distribution(*dims) :
    # Normal distribution
    if opt.data_dist == 'normal' :
        x = torch.randn(*dims)
        if opt.dataset == 'cifar10' :
            x = x * 0.22 + 0.45
        return x
    # Uniform distribution
    elif opt.data_dist == 'uniform' :
        return torch.rand(*dims)
    # Bernoulli distribution with fixed p = 0.5
    elif opt.data_dist == 'bernoulli' :
        p = .5
        return torch.bernoulli(p * torch.ones(*dims))
    # Bernoulli distribution with p varying between 0.05 and 0.95 for every image
    elif opt.data_dist == 'varying_bernoulli' :
        p = torch.rand(dims[0]) * .9 + .05
        p = p.view([-1] + [1] * (len(dims) - 1))
        return torch.bernoulli(p * torch.ones(*dims))
    else :
        raise NotImplementedError(opt.data_dist)

criterion = nn.CrossEntropyLoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # outputs = netD(inputs)
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, inputs)
        else:
            outputs = original_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('Accuracy of the network on netD: %.2f %%' %
            (100. * correct_netD.float() / total))

################################################
# estimate the attack success rate of initial D:
################################################
correct_ghost = 0.0
total = 0.0
netD.eval()
for data in testloader:
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()

    adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
    with torch.no_grad():
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, adv_inputs_ghost)
        else:
            outputs = original_net(adv_inputs_ghost)
            _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct_ghost += (predicted == labels).sum()
print('Attack success rate: %.2f %%' %
        (100 - 100. * correct_ghost.float() / total))
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

# Image dimensions
dims = next(iter(testloader))[0].size()[1:]

batch_num = 1000
best_accuracy = 0.0
best_att = 0.0
for epoch in range(opt.niter):
    netD.train()

    for ii in range(batch_num):
        netD.zero_grad()

        ############################
        # (1) Update D network:
        ###########################
        # Generate random data
        data = sample_random_distribution(opt.batchSize, *dims).to(device)

        # obtain the output label of T
        with torch.no_grad():
            # outputs = original_net(data)
            if opt.dataset == 'azure':
                outputs = cal_azure_proba(clf, data)
                label = cal_azure(clf, data)
            else:
                outputs = original_net(data)
                _, label = torch.max(outputs.data, 1)
                outputs = F.softmax(outputs, dim=1)
            # _, label = torch.max(outputs.data, 1)
        # print(label)

        output = netD(data.detach())
        prob = F.softmax(output, dim=1)
        # print(torch.sum(outputs) / 500.)
        errD_prob = mse_loss(prob, outputs, reduction='mean')
        errD_fake = criterion(output, label) * (1 - opt.beta) + errD_prob * opt.beta
        D_G_z1 = errD_fake.mean().item()
        errD_fake.backward()

        errD = errD_fake
        optimizerD.step()

        del output, errD_fake

        if (ii % 40) == 0:
            print('[%d/%d][%d/%d] D: %.4f D_prob: %.4f D(G(z)): %.4f'
                % (epoch, opt.niter, ii, batch_num,
                    errD.item(), errD_prob.item(), D_G_z1))


    ################################################
    # estimate the attack success rate of trained D:
    ################################################
    correct_ghost = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
        with torch.no_grad():
            # outputs = original_net(adv_inputs_ghost)
            if opt.dataset == 'azure':
                predicted = cal_azure(clf, adv_inputs_ghost)
            else:
                outputs = original_net(adv_inputs_ghost)
                _, predicted = torch.max(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct_ghost += (predicted == labels).sum()
    print('Attack success rate: %.2f %%' %
            (100 - 100. * correct_ghost.float() / total))
    if best_att < (total - correct_ghost):
        torch.save(netD.state_dict(),
                    opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
        best_att = (total - correct_ghost)
        print('This is the best model')
    # worksheet.write(epoch, 0, (correct_ghost.float() / total).item())
    del inputs, labels, adv_inputs_ghost
    torch.cuda.empty_cache()
    gc.collect()

    ################################################
    # evaluate the accuracy of trained D:
    ################################################
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        netD.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))
        if best_accuracy < correct_netD:
            torch.save(netD.state_dict(),
                       opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD
            print('This is the best model')
#     worksheet.write(epoch, 1, (correct_netD.float() / total).item())
# workbook.save('imitation_network_saved_azure.xls')

