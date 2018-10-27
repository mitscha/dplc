from __future__ import print_function
import argparse
import os
import random
import torch


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor

import numpy as np

import math
from models import _netE, _netG, _netDz, _netDim

import logging
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--testroot', default=None, help='uses testing set specified by the path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=512)
parser.add_argument('--ndl', type=int, default=5)
parser.add_argument('--kernel', type=int, default=4, help='kernel size')
parser.add_argument('--nquant', type=int, default=2, help='number of quantization levels')
parser.add_argument('--ncenc', type=int, default=8, help='dimensionality (channels) of quantized representation')
parser.add_argument('--nresenc', type=int, default=0, help='number of residual units at encoder')
parser.add_argument('--nresdec', type=int, default=0, help='number of residual units at decoder/generator')
### OPERATION MODES
parser.add_argument('--useenc', action='store_true', help='use encoder')
parser.add_argument('--freezedec', action='store_true', help='freeze generator/decoder')
parser.add_argument('--comp', action='store_true', help='quantize encoder')
parser.add_argument('--detenc', action='store_true', help='deterministic encoder')
parser.add_argument('--recloss', action='store_true', help='reconstruction loss on output')
parser.add_argument('--wganloss', action='store_true', help='WGAN-GP loss on output')
parser.add_argument('--ncritic', type=int, default=5, help='number of iterations for WGAN critic')
parser.add_argument('--useencdist', action='store_true', help='use encoder distribution for WGAN')
parser.add_argument('--upencwgan', action='store_true', help='update encoder wrt WGAN loss if --useencdist')
parser.add_argument('--intencprior', action='store_true', help='use random interpolates between the encoder \
                                        output and the prior to train the WGAN discriminator, if --useencdist')
###
parser.add_argument('--sigmasqz', type=float, default=1.0, help='variance of prior')
parser.add_argument('--avbtrick', action='store_true', help='trick from AVB paper')
parser.add_argument('--lbd', type=float, default=10.0, help='gan penalty coefficient')
parser.add_argument('--mmd', action='store_true', help='use MMD with IMQ kernel instead of GAN on z-space')
parser.add_argument('--bnz', action='store_true', help='batch-normalize encoder output to zero mean and variance sigmasqz')
parser.add_argument('--lbd_di', type=float, default=1.0, help='image gan loss coefficient')
parser.add_argument('--lbd_gp', type=float, default=10.0, help='coefficient for gradient penalty')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr_dz', type=float, default=0.0005, help='learning rate latent discriminator, default=0.0005')
parser.add_argument('--lr_eg', type=float, default=0.001, help='learning rate encoder-generator, default=0.001')
parser.add_argument('--lr_di', type=float, default=0.0001, help='learning rate image discriminator, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='ADAM beta1, default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2, default=0.999')
parser.add_argument('--wd_di', type=float, default=0.0, help='weight decay for the image discriminator')
parser.add_argument('--wd_eg', type=float, default=0.0, help='weight decay for the WAE encoder and generator')
parser.add_argument('--decay_steps', type=int, nargs='+', default=None, help='when to decay the learning rate')
parser.add_argument('--decay_gamma', type=float, default=0.1, help='decay factor for learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netE', default='', help="path to netG (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netDz', default='', help="path to netDz (to continue training)")
parser.add_argument('--netDim', default='', help="path to netDim (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--check_every', type=int, default=25, help='checkpoint every x epochs')
parser.add_argument('--test_every', type=int, default=5, help='test every x epochs')
parser.add_argument('--vis_every', type=int, default=1000, help='visualize every x iterations')
parser.add_argument('--addreconst', type=int, default=2, help='additional reconstructions to be stored (illustrate random encoder)')
parser.add_argument('--addsamples', type=int, default=0, help='additional samples to be stored (illustrate random encoder)')
parser.add_argument('--lsun_custom_split', action='store_true', help='custom split of LSUN training set (10k testing samples)')
parser.add_argument('--heinit', action='store_true', help='He initialization')
parser.add_argument('--n_samples_var_est', type=int, default=100, help='number of samples to estimate conditional variance if not detenc')



opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('%s/info.log' % opt.outf))
print = logger.info


tb_writer = SummaryWriter(log_dir=opt.outf)

print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: %i"%opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

img_width = opt.imageSize
img_height = opt.imageSize

# data set preparation

if opt.dataset in ['imagenet', 'folder', 'lfw', 'celeba', 'celebahq']:
    # folder dataset
    data_transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=data_transform)
    if opt.testroot is not None:
        testset = dset.ImageFolder(root=opt.testroot,
                                   transform=data_transform)
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
elif opt.dataset == 'lsun':
    data_transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=data_transform)
    # create custom testing set of 10k images if requested
    if opt.lsun_custom_split:
        indices = [i for i in range(3033042)]
        random.shuffle(indices)
        test_indices = indices[:10000]
        train_indices = indices[10000:]
    else:
        opt.testroot = ''
        testset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_val'],
                            transform=data_transform)
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())

elif opt.dataset == 'cityscapes':
    img_width = 2*img_height
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((img_height, img_width)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
assert dataset
if opt.dataset == 'lsun' and opt.lsun_custom_split:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                                         shuffle=False, num_workers=int(opt.workers), drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(test_indices),
                                             shuffle=False, num_workers=int(opt.workers), drop_last=False)
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)

    if opt.testroot is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                                 shuffle=False, num_workers=int(opt.workers), drop_last=False)

if opt.testroot is not None or (opt.dataset == 'lsun' and opt.lsun_custom_split):
    try:
        os.makedirs(opt.outf + '/test')
        os.makedirs(opt.outf + '/test/test_rec')
        if opt.addsamples > 0:
            os.makedirs(opt.outf + '/test/test_samples')
    except OSError:
        pass

# set up parameters
if opt.kernel < 0:
    kernel = 1
else:
    kernel = opt.kernel
padding = (kernel-1)//2
output_padding = 0 if kernel%2 == 0 else 1

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
ndl = int(opt.ndl)

sigmasqz = opt.sigmasqz

nc = 1 if opt.dataset == 'mnist' else 3

ncenc = int(opt.ncenc) if opt.ncenc < opt.nz else opt.nz
if ncenc < 1: ncenc = 1

# custom weights initialization called on netG and netD
if opt.heinit:
    import torch.nn.init as init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_uniform(m.weight.data)
        elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
else:
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            if m.weight is not None:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)



# encoder
quant_delta = 2.0/int(opt.nquant)
quant_levels = [-1.0 + quant_delta/2.0 + l*quant_delta for l in range(opt.nquant)]

if opt.useenc:
    netE = _netE(nc, nz, ngf, kernel, padding, img_width, img_height, quant_levels, opt.comp,  opt.ncenc, opt.nresenc, opt.detenc, quant_delta/2, opt.bnz, ngpu)
    netE.apply(weights_init)

    if opt.netE != '':
        netE.load_state_dict(torch.load(opt.netE))
    print(netE)

# decoder/generator
netG = _netG(nc, nz, ngf, kernel, padding, output_padding, img_width, img_height, opt.nresdec, ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# discriminator on z space
if opt.lbd > 0.0 and not opt.mmd:
    netDz = _netDz(nz, ndf, ndl, ngpu, opt.avbtrick, opt.sigmasqz)
    netDz.apply(weights_init)
    if opt.netDz != '':
        netDz.load_state_dict(torch.load(opt.netDz))
    print(netDz)

# discriminator on image space
if opt.wganloss:
    netDim = _netDim(nc, ngf, kernel, padding, img_width, img_height, ngpu)
    netDim.apply(weights_init)
    if opt.netDz != '':
        netDim.load_state_dict(torch.load(opt.netDim))
    print(netDim)


# losses and variables

# reconstruction loss
reconstruction_loss = nn.MSELoss()
# cross entropy loss for discriminator
criterion = nn.BCEWithLogitsLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = math.sqrt(sigmasqz) * torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_()
label = torch.FloatTensor(opt.batchSize)
real_label_z = 1
fake_label_z = 0


if opt.cuda:
    netG.cuda()
    if opt.lbd > 0.0 and not opt.mmd:
        netDz.cuda()
    if opt.useenc:
        netE.cuda()
    if opt.wganloss:
        netDim.cuda()
    reconstruction_loss.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)


# set up optimizers
if opt.lbd > 0.0 and not opt.mmd:
    optimizerDz = optim.Adam(netDz.parameters(), lr=opt.lr_dz, betas=(opt.beta1, opt.beta2))

if opt.useenc:
    if opt.freezedec:
        optimizerEG = optim.Adam(netE.parameters(), lr=opt.lr_eg, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_eg)
    else:
        optimizerEG = optim.Adam(list(netE.parameters()) + list(netG.parameters()), lr=opt.lr_eg, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_eg)
else:
    optimizerEG = optim.Adam(netG.parameters(), lr=opt.lr_eg, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_eg)


if opt.wganloss:
    optimizerDim = optim.Adam(netDim.parameters(), lr=opt.lr_di, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_di)

# set up learning rate scheduler
if opt.decay_steps is not None:
    if opt.lbd > 0 and not opt.mmd:
        schedDz = MultiStepLR(optimizerDz, milestones=opt.decay_steps, gamma=opt.decay_gamma)
    if opt.wganloss:
        schedDim = MultiStepLR(optimizerDim, milestones=opt.decay_steps, gamma=opt.decay_gamma)
    schedEG = MultiStepLR(optimizerEG, milestones=opt.decay_steps, gamma=opt.decay_gamma)

lbd = opt.lbd
lbd_di = opt.lbd_di

errDz, err_rec, err_gan_z, err_gan_img, Dz_x, Dz_G_z1, Dz_G_z2, gp_wgan, disc_wgan = 0, 0, 0, 0, 0, 0, 0, 0, 0



iter_divisor = opt.ncritic if opt.wganloss else 1

# Calculates the gradient penalty loss for WGAN GP
# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# (modified alpha sampling)
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1))
    alpha = alpha.expand(real_samples.shape)
    if opt.cuda:
        alpha = alpha.cuda()

    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)

    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    if opt.cuda:
        fake = fake.cuda()

    # Get gradient w.r.t. interpolates
    # import pdb; pdb.set_trace()
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lbd_gp
    return gradient_penalty

## Calculates the MMD, sweeping the kernel parameter
## Adapted from
## https://github.com/tolstikhin/wae/blob/eeacd8967b8a8c42ef4d124f3f53e0a7a758924d/wae.py
def compute_mmd_penalty(sample_pz, sample_qz):
    n = opt.batchSize
    sp = sample_pz.view(n, -1)
    sq = sample_qz.view(n, -1)

    norms_pz = torch.sum(torch.pow(sp, 2), dim=1, keepdim=True)
    dotprods_pz = torch.matmul(sp, sp.transpose(0, 1))
    distances_pz = norms_pz + norms_pz.transpose(0,1) - 2. * dotprods_pz

    norms_qz = torch.sum(torch.pow(sq, 2), dim=1, keepdim=True)
    dotprods_qz = torch.matmul(sq, sq.transpose(0, 1))
    distances_qz = norms_qz + norms_qz.transpose(0,1) - 2. * dotprods_qz

    dotprods = torch.matmul(sq, sp.transpose(0,1))
    distances = norms_qz + norms_pz.transpose(0,1) - 2. * dotprods

    Cbase = 2. * nz * sigmasqz

    id = torch.eye(n)
    if opt.cuda:
        id = id.cuda()

    id = Variable(id)

    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz)
        res1 = res1 + C / (C + distances_pz)
        res1 = res1 * (1. - id)
        res1 = torch.sum(res1) / (n * n - n)
        res2 = C / (C + distances)
        res2 = torch.sum(res2) * 2. / (n * n)
        stat = stat + res1 - res2

    return stat

niter = 0

# training loop
for epoch in range(opt.niter):

    if opt.decay_steps is not None:
        if lbd > 0.0 and not opt.mmd:
            schedDz.step()
        if opt.wganloss:
            schedDim.step()
        schedEG.step()

    data_iter = iter(dataloader)

    for i in range(len(data_iter) // iter_divisor):

        # train WGAN discriminator on image space
        if opt.wganloss:
            for _ in range(iter_divisor):
                data, _ = data_iter.next()
                batch_size = data.size(0)
                if opt.cuda:
                    data = data.cuda()
                input.resize_as_(data).copy_(data)
                inputv = Variable(input)

                netDim.zero_grad()

                if opt.useenc and opt.useencdist:

                    if opt.intencprior:
                        ####
                        outE = netE(inputv)
                        outE = outE.detach()
                        alpha = torch.rand((opt.batchSize, 1, 1, 1))
                        alpha = alpha.expand(outE.shape)
                        if opt.cuda:
                            alpha = alpha.cuda()

                        noise.resize_(batch_size, nz, 1, 1).normal_()
                        noise = math.sqrt(sigmasqz) * noise

                        # Get random interpolation between real and fake samples
                        inputG = (alpha * outE + (1 - alpha) * Variable(noise))
                        ####
                    else:
                        inputG = netE(inputv)
                        inputG = inputG.detach()
                else:
                    noise.resize_(batch_size, nz, 1, 1).normal_()
                    noise = math.sqrt(sigmasqz) * noise
                    inputG = Variable(noise)

                # Generate a batch of images
                fake_imgs = netG(inputG)

                # Train on real images
                Dwgan_real = netDim(inputv).mean()

                # Train on fake images
                Dwgan_fake = netDim(fake_imgs.detach()).mean()

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(netDim, inputv.data, fake_imgs.data)

                Dim_loss = Dwgan_fake - Dwgan_real + gradient_penalty

                disc_wgan = Dim_loss.item()
                gp_wgan = gradient_penalty.item()
                Dim_loss.backward()
                optimizerDim.step()

                netDim.zero_grad()

        else:
            data, _ = data_iter.next()
            batch_size = data.size(0)
            if opt.cuda:
                data = data.cuda()
            input.resize_as_(data).copy_(data)
            inputv = Variable(input)


        outputE = None
        lossEG = 0

        if opt.useenc:

            outputE = netE(inputv)

            if opt.lbd > 0.0:
                noise.resize_(batch_size, nz, 1, 1).normal_()
                noise = math.sqrt(sigmasqz) * noise
                if opt.mmd:
                    lossEG = lbd * compute_mmd_penalty(noise, outputE)
                else:
                    # train vanilla GAN discriminator on z space

                    # train with real
                    netDz.zero_grad()
                    label.resize_(batch_size).fill_(real_label_z)
                    labelv = Variable(label)
                    noisev = Variable(noise)

                    outputDz = netDz(noisev.view([batch_size, nz]))
                    errDz_real = criterion(outputDz, labelv)
                    errDz_real.backward()
                    Dz_G_z2 = outputDz.data.mean()
                    Dz_x = outputDz.data.mean()

                    # train with fake
                    labelv = Variable(label.fill_(fake_label_z))
                    outputDz = netDz(outputE.detach().view([batch_size, nz]))
                    errDz_fake = criterion(outputDz, labelv)
                    errDz_fake.backward()
                    Dz_G_z1 = outputDz.data.mean()
                    errDz = errDz_real.item() + errDz_fake.item()
                    optimizerDz.step()

                    # compute gan penlaty on encoder output (z space)
                    labelv = Variable(label.fill_(real_label_z))
                    loss_gan = criterion(netDz(outputE.view([batch_size, nz])), labelv)
                    lossEG = lbd * loss_gan / batch_size

                err_gan_z = lossEG.item()

            netE.zero_grad()

        netG.zero_grad()
        if opt.recloss and opt.useenc:
            outputG = netG(outputE)
            loss_rec = torch.sqrt(reconstruction_loss(outputG, inputv))
            err_rec = loss_rec.item()
            lossEG = lossEG + loss_rec

        if opt.wganloss:
            if opt.useencdist:
                if not opt.upencwgan:
                    # no gradients from WGAN discriminator to encoder
                    outputE = outputE.detach()

            else:
                outputE = Variable(math.sqrt(sigmasqz) * Tensor(batch_size, nz, 1, 1).normal_())
                if opt.cuda:
                    outputE = outputE.cuda()

            loss_gan_img = netDim(netG(outputE)).mean()

            err_gan_img = - lbd_di * loss_gan_img.item()
            lossEG = lossEG - lbd_di * loss_gan_img

        lossEG.backward()
        optimizerEG.step()





        ## do printing, logging etc
        if i % 10 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Rec_loss: %.4f Gan_penalty_z: %.4f Gan_penalty_im: %.4f D(x): %.4f D(G(z)): %.4f / %.4f D_WGAN: %.4f GP_WGAN: %.4f'
                  % (epoch, opt.niter, i*iter_divisor, len(dataloader),
                     errDz, err_rec, err_gan_z, err_gan_img, Dz_x, Dz_G_z1, Dz_G_z2, disc_wgan, gp_wgan))
            tb_writer.add_scalar('data/loss_rec', err_rec, niter)
            tb_writer.add_scalar('data/loss_gan_z', err_gan_z, niter)
            tb_writer.add_scalar('data/loss_gan_im', err_gan_img, niter)
            tb_writer.add_scalar('data/wgan_loss', disc_wgan, niter)
            tb_writer.add_scalar('data/wgan_gp', gp_wgan, niter)
            tb_writer.add_scalar('data/gan_z_loss', errDz, niter)
            tb_writer.add_scalar('data/gan_z_dx', Dz_x, niter)
            tb_writer.add_scalars('data/gan_z_dgz', {'r': Dz_G_z1, 'f': Dz_G_z2}, niter)
        if i % opt.vis_every == 0:
            vutils.save_image(inputv.data,
                    '%s/real_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
            if opt.useenc:
                reconst = netG(netE(inputv))
                vutils.save_image(reconst.data,
                        '%s/reconstruction_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                if opt.comp:
                    reconst = netG(netE(inputv))
                    vutils.save_image(reconst.data,
                            '%s/reconstruction2_epoch_%03d.png' % (opt.outf, epoch),
                            normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_fixed_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        niter += 1

    ## do testing, logging etc
    if (epoch % opt.test_every == 0 or epoch == opt.niter-1):
        if (opt.testroot is not None and opt.useenc):
            testloss = 0
            nbatch = 0
            nimg = 0
            for i, data in enumerate(testloader, 0):
                real_cpu, _ = data
                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                reconst = netG(netE(inputv))

                testloss += reconstruction_loss(reconst, inputv).item()

                imin = reconst.data.min()
                imax = reconst.data.max()
                for j, img in enumerate(reconst.data, opt.batchSize*nbatch):
                    vutils.save_image((img - imin)/(imax - imin), '%s/test/test_rec/img_%05d.png' % (opt.outf, j))

                nbatch += 1

                if epoch == 0:
                    vutils.save_image(inputv.data,
                            '%s/test/real_samples_epoch_%03d_batch_%03d.png' % (opt.outf, epoch, i), normalize=True)
                vutils.save_image(reconst.data,
                        '%s/test/reconstruction_epoch_%03d_batch_%03d.png' % (opt.outf, epoch, i), normalize=True)
                reconst = netG(netE(inputv))
                vutils.save_image(reconst.data,
                        '%s/test/reconstruction2_epoch_%03d_batch_%03d.png' % (opt.outf, epoch, i), normalize=True)
                for j in range(3, 3+opt.addreconst):
                    reconst = netG(netE(inputv))
                    vutils.save_image(reconst.data,
                            '%s/test/reconstruction%02d_batch_%03d.png' % (opt.outf, j, i), normalize=True)
                if not opt.detenc and i == 0:
                    batch_mean = torch.zeros_like(inputv)
                    batch_mean_sq = torch.zeros_like(inputv)
                    for j in range(opt.n_samples_var_est):
                        cur_reconst = netG(netE(inputv.detach())).data
                        batch_mean = batch_mean + cur_reconst
                        batch_mean_sq = batch_mean_sq + cur_reconst**2

                    cond_var = torch.mean(batch_mean_sq/opt.n_samples_var_est - (batch_mean/opt.n_samples_var_est)**2)
                    print('Reconstruction conditional variance: %.4f'%(cond_var.item()))

            print('Testing reconstruction MSE: %.4f'%(testloss/nbatch))
            tb_writer.add_scalar('data/loss_rec_test', testloss/nbatch, niter)

        if opt.addsamples > 0:
            for i in range(opt.addsamples//opt.batchSize):
                noise.resize_(batch_size, nz, 1, 1).normal_()
                noise = math.sqrt(sigmasqz) * noise
                inputG = Variable(noise)
                samples = netG(inputG)

                imin = samples.data.min()
                imax = samples.data.max()
                for j, img in enumerate(samples.data, i*opt.batchSize):
                    vutils.save_image((img - imin)/(imax - imin), '%s/test/test_samples/img_%05d.png' % (opt.outf, j))



    # do checkpointing
    if (epoch % opt.check_every == 0 or epoch == opt.niter-1) and not epoch == 0:
        if opt.useenc:
            torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        if opt.lbd > 0 and not opt.mmd:
            torch.save(netDz.state_dict(), '%s/netDz_epoch_%d.pth' % (opt.outf, epoch))
        if opt.wganloss > 0:
            torch.save(netDim.state_dict(), '%s/netDim_epoch_%d.pth' % (opt.outf, epoch))

tb_writer.close()
