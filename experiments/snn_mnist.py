import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
import math
import datetime
import sys
import torch.nn.functional as F


class PoissonGenerator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input) * 1.0).float(), torch.sign(input))
        return out


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad

class SNN_Model(nn.Module):
    def __init__(self, labels=10, kernel_size=3, dropout=0.2, default_threshold=1.0, timesteps=100):
        super(SNN_Model, self).__init__()
        self.act_func = SurrGradSpike.apply
        self.input_layer = PoissonGenerator()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.labels = labels
        self.timesteps = timesteps
        self.threshold = {}
        self.mem = {}
        self.mask = {}
        self.spike = {}
        self.leak = torch.tensor(1.0)

        self.features, self.classifier = self._make_layers()
        self._initialize_weights2()

        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.threshold[l] = torch.tensor(default_threshold)

        prev = len(self.features)
        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                self.threshold[prev + l] = torch.tensor(default_threshold)

    def _make_layers(self):
        layers = []
        stride = 1
        in_channels = 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                             stride=stride, bias=False),
                   nn.ReLU(inplace=True)
                   ]
        in_channels = 32
        layers.pop()
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(in_channels, 64, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                             stride=stride, bias=False),
                   nn.ReLU(inplace=True)
                   ]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        features = nn.Sequential(*layers)

        layers = []

        layers += [nn.Linear(64 * 7 * 7, 512, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(512, 512, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(512, self.labels, bias=False)]

        classifier = nn.Sequential(*layers)
        return features, classifier

    def _initialize_weights2(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def network_update(self, timesteps, leak):
        self.timesteps = timesteps
        self.leak = torch.tensor(leak)

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)

        self.mem = {}
        self.mask = {}
        self.spike = {}

        for l in range(len(self.features)):

            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)

            elif isinstance(self.features[l], nn.Dropout):
                self.mask[l] = self.features[l](torch.ones(self.mem[l - 2].shape))

            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width // self.features[l].kernel_size
                self.height = self.height // self.features[l].kernel_size

        prev = len(self.features)

        for l in range(len(self.classifier)):

            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)

            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev + l] = self.classifier[l](torch.ones(self.mem[prev + l - 2].shape))

        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)

    def forward(self, x):
        self.neuron_init(x)
        max_mem = 0.0

        for t in range(self.timesteps):
            out_prev = self.input_layer(x)

            for l in range(len(self.features)):

                if isinstance(self.features[l], (nn.Conv2d)):

                    mem_thr = (self.mem[l] / self.threshold[l]) - 1.0
                    # out = self.act_func(mem_thr, (t - 1 - self.spike[l]))
                    out = self.act_func(mem_thr)
                    rst = self.threshold[l] * (mem_thr > 0).float()
                    self.spike[l] = self.spike[l].masked_fill(out.bool(), t - 1)
                    self.mem[l] = self.leak * self.mem[l] + self.features[l](out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev = self.features[l](out_prev)

                elif isinstance(self.features[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            out_prev = out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)

            for l in range(len(self.classifier) - 1):
                if isinstance(self.classifier[l], (nn.Linear)):

                    mem_thr = (self.mem[prev + l] / self.threshold[prev + l]) - 1.0
                    # out = self.act_func(mem_thr, (t - 1 - self.spike[prev + l]))
                    out = self.act_func(mem_thr)
                    rst = self.threshold[prev + l] * (mem_thr > 0).float()
                    self.spike[prev + l] = self.spike[prev + l].masked_fill(out.bool(), t - 1)

                    self.mem[prev + l] = self.leak * self.mem[prev + l] + self.classifier[l](out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[prev + l]

            # Compute the classification layer outputs
            self.mem[prev + l + 1] = self.mem[prev + l + 1] + self.classifier[l + 1](out_prev)

        return self.mem[prev + l + 1]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(epoch):
    global learning_rate

    # model.module.network_update(timesteps=timesteps, leak=leak)
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    # if epoch in lr_interval:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] / lr_reduce
    #         learning_rate = param_group['lr']

    # f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))

    # total_loss = 0.0
    # total_correct = 0
    model.train()

    # current_time = start_time
    # model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        # pdb.set_trace()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item() / data.size(0), data.size(0))

        if (batch_idx + 1) % train_acc_batches == 0:
            temp1 = []
            for value in model.threshold.values():
                temp1 = temp1 + [round(value.item(), 2)]
            f.write(
                '\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}'
                .format(epoch,
                        batch_idx + 1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        model.leak.item(),
                        model.timesteps
                        )
                )
    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'
            .format(epoch,
                    learning_rate,
                    losses.avg,
                    top1.avg,
                    )
            )


def test(epoch):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy

        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(), data.size(0))
            top1.update(correct.item() / data.size(0), data.size(0))

            if test_acc_every_batch:
                f.write('\nAccuracy: {}/{}({:.4f})'
                    .format(
                    correct.item(),
                    data.size(0),
                    top1.avg
                )
                )

        temp1 = []
        for value in model.threshold.values():
            temp1 = temp1 + [value.item()]

        if epoch > 5 and top1.avg < 0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg > max_accuracy:
            max_accuracy = top1.avg

            state = {
                'accuracy': max_accuracy,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'thresholds': temp1,
                'timesteps': timesteps,
                'leak': leak,
            }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass
            filename = './trained_models/snn/' + identifier + '.pth'
            torch.save(state, filename)

            # if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg,
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        )
        )


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    f = sys.stdout
    timesteps = 100
    leak = 1.0
    identifier = 'snn_' + str(timesteps)
    test_acc_every_batch = False
    train_acc_batches = 100
    learning_rate = 1e-4
    trainset = datasets.MNIST(root='../Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(root='../Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
    labels = 10

    batch_size = 32

    model = SNN_Model(timesteps=timesteps)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True, weight_decay=5e-4)


    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    f.write('\n {}'.format(optimizer))
    max_accuracy = 0

    # print(model)
    # f.write('\n Threshold: {}'.format(model.module.threshold))

    for epoch in range(1, 10):
        start_time = datetime.datetime.now()

        train(epoch)
        test(epoch)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))