import torch
from torch import nn
from resblock import BasicBlock
from torch.autograd import Variable
from scalar_quantizer import quantize
import math



# Encoder and stochastic function (B in the paper)
class _netE(nn.Module):
    def __init__(self, nc, nz, ngf, kernel=2, padding=1, img_width=64, img_height=64,
                    quant_levels=None, do_comp=False, ncenc=8, nresenc=0, detenc=False,
                    noisedelta=0.5, bnz=False, ngpu=1):
        super(_netE, self).__init__()
        self.ngpu = ngpu
        self.detenc = detenc or not do_comp
        self.noisedelta = noisedelta
        self.nfmodelz = math.ceil(nz / ((img_height//16) * (img_width//16))) + ncenc
        self.ncenc = ncenc

        model_down_list = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngf, kernel, 2, padding, bias=False),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ngf, ngf * 2, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        ]
        # state size. (ndf*8) x 4 x 4

        # quantize if in compression mode
        if do_comp:
            model_down_list += [
                nn.Conv2d(ngf * 8, ncenc, 3, 1, 1, bias=True),
                quantize(quant_levels)
            ]

        self.model_down = nn.Sequential(*model_down_list)

        # stochastic function mapping compressed representation to latent space
        # of generator (B in paper)
        if do_comp:
            model_z_list = [
                nn.ConvTranspose2d(ncenc, ngf * 8, 3, 1, 1, bias=True) if detenc \
                    else nn.ConvTranspose2d(self.nfmodelz, ngf * 8, 3, 1, 1, bias=True)
            ]
        else:
            model_z_list = []

        if nresenc > 0:
            model_z_list += [BasicBlock(ngf * 8, ngf * 8) for _ in range(nresenc)]

        model_z_list += [nn.Conv2d(ngf * 8, nz, (img_height//16, img_width//16), 1, 0, bias=False)]

        # batchnorm to facilitate prior matching
        if bnz:
            model_z_list += [nn.BatchNorm2d(nz)]

        self.model_z = nn.Sequential(*model_z_list)


    def forward(self, input):
        use_cuda = isinstance(input.data, torch.cuda.FloatTensor)
        if use_cuda and self.ngpu > 1:
            out_down = nn.parallel.data_parallel(self.model_down, input, range(self.ngpu))
        else:
            out_down = self.model_down(input)

        if not self.detenc:
            # feed noise of appropriate dimension when using stoc. function
            out_down_pad_size = list(out_down.size())
            out_down_pad_size[1] = self.nfmodelz - self.ncenc
            out_down_pad = torch.zeros(out_down_pad_size)
            out_down_pad.uniform_(-self.noisedelta, self.noisedelta)
            if use_cuda:
                out_down_pad = out_down_pad.cuda()
            out_down = torch.cat([out_down, Variable(out_down_pad)], 1)

        if use_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model_z, out_down, range(self.ngpu))
        else:
            output = self.model_z(out_down)

        return output


# Standard DCGAN-type generator/decoder
class _netG(nn.Module):
    def __init__(self, nc, nz, ngf, kernel=2, padding=1, output_padding=0, img_width=64, img_height=64, nresdec=0, ngpu=1):
        super(_netG, self).__init__()
        self.ngpu = ngpu

        # input is z, going into a convolution
        main_list = [nn.ConvTranspose2d(nz, ngf * 8, (img_height//16, img_width//16), 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True)]

        if nresdec > 0:
            main_list += [BasicBlock(ngf * 8, ngf * 8) for _ in range(nresdec)]

        main_list += [
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel, 2, padding, output_padding, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel, 2, padding, output_padding, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, kernel, 2, padding, output_padding, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      ngf, kernel, 2, padding,  output_padding, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(    ngf,      nc, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        ]

        self.main = nn.Sequential(*main_list)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


# MLP discriminator in z-space
class _netDz(nn.Module):
    def __init__(self, nz, ndf=512, ndl=5, ngpu=0, avbtrick=False, sigmasq=1):
        super(_netDz, self).__init__()
        self.ngpu = ngpu
        self.avbtrick = avbtrick
        self.sigmasqz = sigmasq
        self.nz = nz

        layers = [[nn.Linear(ndf, ndf), nn.ReLU(True)] for _ in range(ndl-2)]

        layers = [nn.Linear(nz, ndf), nn.ReLU(True)] \
                    + sum(layers, []) \
                    + [nn.Linear(ndf, 1)]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # Nowozin trick from WAE paper, only valid for Gaussian prior
        if self.avbtrick:
            output = output - torch.norm(input, p=2, dim=1, keepdim=True)**2 / 2 / self.sigmasqz \
                        - 0.5 * math.log(2 * math.pi) \
                        - 0.5 * self.nz * math.log(self.sigmasqz)

        return output.view(-1, 1).squeeze(1)


# DCGAN-style discriminator in image space
class _netDim(nn.Module):
    def __init__(self, nc=3, ndf=64, kernel=2, padding=1, img_width=64, img_height=64, ngpu=1):
        super(_netDim, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(    nc,      ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf, img_height//2, img_width//2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 2, img_height//4, img_width//4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 4, img_height//8, img_width//8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 8, img_height//16, img_width//16]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, (img_height//16, img_width//16), 1, 0, bias=False),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)
