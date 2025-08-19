"""
StyleGAN2 for Surface Generation
================================

This module implements a StyleGAN2 variant for generating synthetic heliostat
surfaces. It is adapted from Lucidrains' stylegan2-pytorch
(https://github.com/lucidrains/stylegan2-pytorch) with modifications
for single-channel (grayscale) surface data and integration with the
PhysConUL pipeline.

References:
- Lewen et al., (https://doi.org/10.48550/arXiv.2408.10802)
- Lucidrains' stylegan2-pytorch repository

Main components:
- Data loading and augmentation utilities
- Generator, Discriminator, and StyleVectorizer definitions
- StyleGAN2 wrapper class with exponential moving average (EMA)
- Trainer class for handling training, evaluation, interpolation, projection
- Utility functions for noise, augmentations, plotting, and FID evaluation

Note:
This implementation is a key component of the thesis pipeline,
used to generate synthetic surfaces and augment empirical datasets.
"""
import argparse
from datetime import datetime
import time
import math
import json
from math import floor, log2
import random 
from shutil import rmtree
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import grad as torch_grad
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from PhysConUL_DownCont.Jan_NN_model import utils
#from vector_quantize_pytorch import VectorQuantize
from einops.einops import einsum, rearrange
#from diffAugm import DiffAugment
import os 
import PhysConUL_DownCont.Jan_NN_model.utils
#from defaults import get_cfg_defaults
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision
from torchvision import transforms 
from torch.optim.lr_scheduler import StepLR
from shutil import rmtree
from Unsupervised_learning.Dataloader_Jan import normalize_between #Dataset_surfaces


EPS = 1e-8

#CLUSTER = True
#cfg = get_cfg_defaults()

try:
    import horovod.torch as hvd
except:
    CLUSTER = False

if not CLUSTER:
    from PIL import Image
    from tqdm import tqdm

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False





# def give_random_surface_from_styleGAN(model, trunc_psi=1, renormalize=True, to_facets=True):
    
#     latent_dim = model.GAN.G.latent_dim
#     image_size = model.GAN.G.image_size
#     num_layers = model.GAN.G.num_layers
    
#     # latents and noise
    
#     latents = [(noise(1, latent_dim), num_layers)]
#     noi = image_noise(1, image_size)
    
#     generated_surface = model.generate_truncated(model.GAN.SE, 
#                                                  model.GAN.GE, 
#                                                  latents, 
#                                                  noi=noi, 
#                                                  trunc_psi=trunc_psi)  
    
#     # generated_surface = self.generator.to_physical_surface(generated_surface,
#     #                                              min_alt, 
#     #                                              max_alt, 
#     #                                              to_facets=True)
        
#     return generated_surface

        
def give_diff_augments():    
    
    types=['brightness', 
           'contrast', 
           'lightcontrast', 
           'saturation', 
           'lightsaturation',
           'color', 
           'lightcolor', 
           'translation', 
           'cutout',
           'crop',
           'offset']
        
    return types

def arg_parse():
    parser = argparse.ArgumentParser()
    
    if CLUSTER:
        parser.add_argument(
            "--data_path", help="data path (NPZ)", type=str,
            default = r'/p/scratch/hai_gancstr/lewen1/cntrl_points'
    
        )
    else:
        parser.add_argument(
            "--data_path", help="data path (NPZ)", type=str,
            default = r'C:\Users\lewe_jn\Desktop\gancstr\rawdata\deflektometrie\cntrl_points'
    
        )
        
    parser.add_argument(
        "--name",
        help="experiment name, results will be saved in results/<name> and models/<name>",
        type=str, default='styleGAN'
    )
    parser.add_argument(
        "--image_size", help="image size", type=int, default=16,
    )
    parser.add_argument(
        "--epochs", help="number of epochs", type=int, default=1000000,
    )
    parser.add_argument(
        "--batch_size", help="batch size", type=int, default=32,
    )
    
    parser.add_argument(
        "--augmentation_types", help="augmentation types", type=list, 
        default=['brightness', 
               'contrast', 
               'lightcontrast', 
               'saturation', 
               'lightsaturation',
               'color', 
               'lightcolor', 
               'translation', 
               'cutout'],
    )
    args = parser.parse_args()
    return args


def plot_grid(batch, batch_size, title='', savename='', normalize=False, 
              save=False):
    
    # img_size = arg_parse().image_size

    fig, ax = plt.subplots(1,1)
    ax.axis("off")
    fig.suptitle(title)
    grid = vutils.make_grid(batch[:batch_size], padding=1, normalize=normalize)
    ax.imshow(np.transpose(grid.cpu(),(1,2,0))[:,:,0], cmap='jet')
        
    plt.show()
    if save:
        fig.savefig(savename)
    plt.close(fig)
    
    
    
def get_dataset_old(path):
    train_images = np.load(path)["images"]
    train_images = train_images.transpose((0, 3, 1, 2))
    train_images = (torch.from_numpy(train_images) / 255.0).float()
    dataset = TensorDataset(train_images)
    return dataset


def get_dataset(path, is_train_or_valid):
    dataset = Dataset_surfaces(cfg, path, is_train_or_valid)
    return dataset

    
def get_dataloader(path, batch_size, is_train_or_valid):
    dataset = get_dataset(path, is_train_or_valid)
    min_alt = float(dataset.minimum)
    max_alt = float(dataset.maximum)
    
    if CLUSTER:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
        # no need for workers, data is fully in memory in this case
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=0, 
            drop_last=True, 
            sampler=sampler,
        )
        
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=0, 
            drop_last=True, 
            shuffle=True
        )
    return dataset, dataloader, min_alt, max_alt


class NanException(Exception):
    pass


class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


def default(value, d):
    return d if value is None else value


def cycle(iterable, sampler):
    epoch = 0
    while True:
        if CLUSTER:
            sampler.set_epoch(epoch)
        for i in iterable:
            yield i
        epoch += 1
        if CLUSTER:
            if hvd.rank() == 0:
                print(f'Epoch {epoch} finished')


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(
        outputs=outputs,
        inputs=styles,
        grad_outputs=torch.ones(outputs.shape).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0.0, 1.0).cuda()


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def evaluate_in_chunks_fromW(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


def convert_rgb_to_transparent(image):
    if image.mode == "RGB":
        return image.convert("RGBA")
    return image


def convert_transparent_to_rgb(image):
    if image.mode == "RGBA":
        return image.convert("RGB")
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f"image with invalid number of channels given {channels}")

        if alpha is None and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def random_float(lo, hi):
    return lo + (hi - lo) * random.random()


def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random.random() * delta)
    w_delta = int(random.random() * delta)
    cropped = tensor[
        :, :, h_delta : (h_delta + new_width), w_delta : (w_delta + new_width)
    ].clone()
    return F.interpolate(cropped, size=(h, h), mode="bilinear")


def random_hflip(tensor, prob):
    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, types, detach=False):
        # if random() < prob:
    
        
        images = DiffAugment(images, p_augment=0, types=types)
        plot_grid(images, 16)
        
        if detach:
            # images.detach_()
            images.detach()

        return self.D(images)


# stylegan2 classes


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        # out_filters = 3 if not rgba else 4
        out_filters = 1
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)
        
        # added by Jan
        # x = torch.sigmoid(x)
        return x


class Conv2DMod(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs
    ):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(
            self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape
        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_channels,
        filters,
        upsample=True,
        upsample_rgb=True,
        rgba=False,
        noise=False
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )
        self.noise = noise
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        if self.noise:
            self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        if self.noise:
            self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)
            
        if self.noise:
            inoise = inoise[:, : x.shape[2], : x.shape[3], :]
            noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
            noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        if self.noise:
            x = self.activation(x + noise1)
        else:
            x = self.activation(x)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        if self.noise:
            x = self.activation(x + noise2)
        else:
            x = self.activation(x)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb




class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1)
        )

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.downsample = (
            nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        network_capacity=16,
        transparent=False,
        attn_layers=[],
        no_const=False,
        fmap_max=128,
        noise=False
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        self.noise=noise
        
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][
            ::-1
        ]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(
                latent_dim, init_channels, 4, 1, 0, bias=False
            )
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
                noise=noise
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

    def to_physical_surface(self, generated_surface, min_alt, max_alt, to_facets=True):
        
        # # min_alt = model.min_alt
        # # max_alt = model.max_alt
        
        # if min_alt == None:
        #     min_alt = -0.003
        #     max_alt = 0.003
            
        generated_surface = normalize_between(generated_surface, 0, 1, min_alt, max_alt)
        
        if to_facets:
            generated_surface = utils.surface_to_facets(generated_surface)
        
        return generated_surface    

class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        network_capacity=16,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        transparent=False,
        fmap_max=128,
    ):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        
        # num_init_filters = 3 if not transparent else 4
        num_init_filters = 1
        
        blocks = []
        
        # filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]
        filters = [num_init_filters] + [(4*network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = (
                PermuteToFrom(VectorQuantize(out_chan, fq_dict_size))
                if num_layer in fq_layers
                else None
            )
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last
        
        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()

        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
                
        b, *_ = x.shape
        
        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(
            self.blocks, self.attn_blocks, self.quantize_blocks
        ):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

            if q_block is not None:
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


class StyleGAN2(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim=128,
        fmap_max=128,
        style_depth=8,
        network_capacity=16,
        transparent=False,
        fp16=False,
        cl_reg=False,
        steps=1,
        lr=1e-4,
        ttur_mult=2,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        lr_mlp=0.1,
        noise=False
    ):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.noise = noise
        
        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.G = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
            fmap_max=fmap_max,
            noise=noise
        )
        self.D = Discriminator(
            image_size,
            network_capacity,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            transparent=transparent,
            fmap_max=fmap_max,
        )

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
            fmap_max=fmap_max,
            noise=noise
        )
        
        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner

            # experimental contrastive loss discriminator regularization
            assert (
                not transparent
            ), "contrastive loss regularization does not work with transparent images yet"
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer="flatten")

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = None
        self.D_opt = None
        
        # self.G_opt = AdamP(generator_params, lr=self.lr, betas=(0.5, 0.9))
        # self.D_opt = AdamP(
        #     self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9)
        # )
        
        # if CLUSTER:
        #     self.G_opt = hvd.DistributedOptimizer(self.G_opt)
        #     self.D_opt = hvd.DistributedOptimizer(self.D_opt)
            
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()

        self.fp16 = fp16
        if fp16:
            (
                (self.S, self.G, self.D, self.SE, self.GE),
                (self.G_opt, self.D_opt),
            ) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE],
                [self.G_opt, self.D_opt],
                opt_level="O1",
                num_losses=3,
            )

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
        
        if self.noise:
            for block in self.G.blocks:
                nn.init.zeros_(block.to_noise1.weight)
                nn.init.zeros_(block.to_noise2.weight)
                nn.init.zeros_(block.to_noise1.bias)
                nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()
            ):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
        
class Trainer:
    def __init__(
        self,
        name,
        results_dir,
        models_dir,
        image_size,
        network_capacity,
        latent_dim=128,
        fmap_max=256,
        style_depth=8,
        transparent=False,
        batch_size=4,
        mixed_prob=0.9,
        gradient_accumulate_every=1,
        lr=2e-4,
        lr_mlp=1.0,
        ttur_mult=2,
        num_workers=None,
        save_every=1000,
        trunc_psi=0.6,
        fp16=False,
        cl_reg=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        aug_prob=0.0,
        dataset_aug_prob=0.0,
        steps_per_epoch=None,
        augmentation_types=None,
        noise=False,
        augment_surfaces_to_mean=False,
        *args,
        **kwargs,
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.fid_dir = results_dir + '/fid' 
        self.config_path = self.models_dir / name / "config.json"
        # self.config_path = None

        assert log2(
            image_size
        ).is_integer(), "image size must be a power of 2 (64, 128, 256, 512, 1024)"
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.style_depth = style_depth
        self.noise=noise
       
        self.transparent = transparent
        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const
        self.aug_prob = aug_prob

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.std = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0
        self.q_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.trainset = None
        self.valid_loader = None
        self.validSet_size = None

        self.dataset_aug_prob = dataset_aug_prob
        self.augmentation_types = augmentation_types
        self.augment_surfaces_to_mean = augment_surfaces_to_mean
        self.steps_per_epoch = steps_per_epoch
        
        self.loss_list_G = []
        self.loss_list_D = []
        self.loss_list_D_real = []
        self.loss_list_D_fake = []
        self.pl_mean_list = []
        self.min_alt = None
        self.max_alt = None
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(
            lr=self.lr,
            lr_mlp=self.lr_mlp,
            ttur_mult=self.ttur_mult,
            image_size=self.image_size,
            network_capacity=self.network_capacity,
            latent_dim=self.latent_dim,
            fmap_max=self.fmap_max,
            style_depth=self.style_depth,
            transparent=self.transparent,
            fq_layers=self.fq_layers,
            fq_dict_size=self.fq_dict_size,
            attn_layers=self.attn_layers,
            fp16=self.fp16,
            cl_reg=self.cl_reg,
            no_const=self.no_const,
            noise=self.noise
            *args,
            **kwargs,
        )
        
        # param_list = list(self.GAN.G.parameters()) + list(self.GAN.S.parameters())

        # print('Generator parameter: ' + str(sum(p.numel() for p in self.GAN.G.parameters() if p.requires_grad)))
        # print('Mapping network parameter: ' + str(sum(p.numel() for p in self.GAN.S.parameters() if p.requires_grad)))
        # print('Discriminator parameter: ' + str(sum(p.numel() for p in self.GAN.D.parameters() if p.requires_grad)))
  
        # print(self.GAN.G)
        # print(self.GAN.D)

        # if CLUSTER:
        #     hvd.broadcast_parameters(self.GAN.state_dict(), root_rank=0)
        #     hvd.broadcast_optimizer_state(self.GAN.G_opt, root_rank=0)
        #     hvd.broadcast_optimizer_state(self.GAN.D_opt, root_rank=0)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = (
            self.config()
            if not self.config_path.exists()
            else json.loads(self.config_path.read_text())
        )
        self.image_size = config["image_size"]
        self.network_capacity = config["network_capacity"]
        self.transparent = config["transparent"]
        self.fq_layers = config["fq_layers"]
        self.fq_dict_size = config["fq_dict_size"]
        self.attn_layers = config.pop("attn_layers", [])
        self.no_const = config.pop("no_const", False)
        self.latent_dim = config["latent_dim"]
        self.fmap_max = config["fmap_max"]
        self.style_depth = config["style_depth"]
        self.aug_prob = config["aug_prob"]
        self.dataset_aug_prob = config["dataset_aug_prob"]
        self.augmentation_types = config["augmentation_types"]

        try:
            self.loss_list_G = config["loss_list_G"]
            self.loss_list_D = config["loss_list_D"]
            self.lr = config["lr"]
            self.lr_mlp = config["lr_mlp"]
            self.ttur_mlp = config["ttur_mlp"]

        except:
            print('------------- logging could not log all stats! -----------------')
        
        try:
            # self.noise = config["noise"]

            self.loss_list_D_real = config["loss_list_D_real"]
            self.loss_list_D_fake = config["loss_list_D_fake"]
            
            # print(self.loss_list_D_fake)
        except:
            print('------------- logging could not log loss_list_D_real and _fake! -----------------')
        
        try:
            self.noise = config["noise"]        
        except:
            print('------------- logging could not log noise! -----------------')
        
        # try:
        #     self.min_neu = config["min_neu"]        
        #     self.min_alt = config["min_alt"]        
        # except:
        #     print('------------- data set infos were not logged -----------------')
        
        
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            # netstructure
            "image_size": self.image_size,
            "network_capacity": self.network_capacity,
            "transparent": self.transparent,
            "fq_layers": self.fq_layers,
            "fq_dict_size": self.fq_dict_size,
            "attn_layers": self.attn_layers,
            "no_const": self.no_const,
            "fmap_max": self.fmap_max,
            "latent_dim": self.latent_dim,
            "style_depth": self.style_depth,
            "noise": self.noise,
            # training paras
            "lr": self.lr,
            "lr_mlp": self.lr_mlp,
            "ttur_mlp": self.ttur_mult,
            "aug_prob": self.aug_prob,
            "dataset_aug_prob": self.dataset_aug_prob,
            "augmentation_types": self.augmentation_types,
            "gradient_accumulate_every": self.gradient_accumulate_every,
            "loss_list_G": self.loss_list_G,
            "loss_list_D":self.loss_list_D,
            "loss_list_D_fake": self.loss_list_D_fake,
            "loss_list_D_real":self.loss_list_D_real,
            # dataset paras
            "min_alt":self.min_alt, 
            "max_alt": self.max_alt
        }

    def set_data_src(self, path):
        trainset, train_loader, min_alt, max_alt = get_dataloader(path, self.batch_size, is_train_or_valid='train')
        self.loader = cycle(train_loader, train_loader.sampler)
        self.trainset = trainset
        self.min_alt = min_alt
        self.max_alt = max_alt
        
    def set_valid_data(self, path):
        _, valid_loader, _, _ = get_dataloader(path, 1, is_train_or_valid='valid')
        self.valid_loader = valid_loader
            
    def train(self):
        assert (
            self.loader is not None
        ), "You must first initialize the data source with `.set_data_src(<folder of images>)`"

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.0).cuda()
        total_gen_loss = torch.tensor(0.0).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)

        if self.GAN.D_cl is not None:
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = (
                        mixed_list if random.random() < self.mixed_prob else noise_list
                    )
                    style = get_latents_fn(batch_size, num_layers, latent_dim)
                    noise = image_noise(batch_size, image_size)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)
                    
                    generated_images = self.GAN.G(w_styles, noise)
                    
                    
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                (image_batch, _, _) = next(self.loader)
                    
                image_batch = image_batch.cuda()
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, 0)

            self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        
        types = self.augmentation_types
        type_list = []
        for t in types:
            if random.random() < aug_prob:
                type_list.append(t)
                
        for i in range(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random.random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)
            
            generated_images = self.GAN.G(w_styles, noise)
            fake_output, fake_q_loss = self.GAN.D_aug(
                generated_images.clone().detach(), detach=True, types=type_list
            )
            
            # fake_output, fake_q_loss = self.GAN.D_aug(
            #     generated_images.clone().detach(), detach=True, prob=aug_prob
            # )
            
            (image_batch, _, _) = next(self.loader)
            
            # plot_grid(image_batch, 16, normalize=True)

            if self.augment_surfaces_to_mean and random.random() < self.dataset_aug_prob:
                idc = torch.randperm(len(self.trainset))
                idc = idc[:self.batch_size]
                images = self.trainset.__getitem__(idc)
                
                weights = torch.rand(self.batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                
                image_batch = torch.mul(weights,image_batch) + torch.mul((torch.ones_like(weights) - weights),images)
                
                
            # plot_grid(images, 16, normalize=True)
            # plot_grid(image_batch, 16, normalize=True)
            
            
            
            image_batch = image_batch.cuda()
            image_batch.requires_grad_()
            real_output, real_q_loss = self.GAN.D_aug(image_batch, types=type_list)
            # real_output, real_q_loss = self.GAN.D_aug(image_batch, prob=aug_prob)


            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            quantize_loss = (fake_q_loss + real_q_loss).mean()
            self.q_loss = float(quantize_loss.detach().item())

            disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, 1)

            total_disc_loss += (
                divergence.detach().item() / self.gradient_accumulate_every
            )
            self.nbims += len(image_batch)

        self.d_loss = float(total_disc_loss)
        self.loss_list_D_real.append(float(F.relu(1 + real_output).mean().detach()))
        self.loss_list_D_fake.append(float(F.relu(1 - fake_output).mean().detach()))
        
        # print(self.loss_list_D_real[-1], self.loss_list_D_fake[-1])
        self.GAN.D_opt.step()

        
        # train generator

        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            
            noise = image_noise(batch_size, image_size)
            
            w_space = latent_to_w(self.GAN.S, style)

            w_styles = styles_def_to_tensor(w_space)
            generated_images = self.GAN.G(w_styles, noise)
                
            fake_output, _ = self.GAN.D_aug(generated_images, types=type_list)
            # fake_output, _ = self.GAN.D_aug(generated_images, prob=aug_prob)
            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()
        
        self.loss_list_G.append(self.g_loss)
        self.loss_list_D.append(self.d_loss)
        
        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(
                f"NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}"
            )
            self.load(checkpoint_num)
            raise NanException

        # periodically save results
        if CLUSTER:
            if (hvd.rank() == 0) and (self.steps % self.save_every == 0):
                self.save(checkpoint_num)
        
            if (hvd.rank() == 0) and (
                self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500)
            ):
                self.evaluate(floor(self.steps / 1000))
        else:
            if (
                self.steps % 1000 == 0 #or (self.steps % 100 == 0 and self.steps < 2500)
            ):
                self.evaluate(floor(self.steps / 1000))
                self.save(checkpoint_num)

        self.steps += 1
        self.av = None
        
    
    @torch.no_grad()
    def plot_loss(self):
        self.GAN.eval()
        
        
        fig, ax = plt.subplots(1,2, sharex='all', sharey='all')
        
        ax[0].plot(self.loss_list_D, label='loss D', alpha=0.5)
        ax[0].plot(self.loss_list_G, label='loss G', alpha=0.5)
        
        ax[1].plot(self.loss_list_D_real, label='loss D real', alpha=0.5)
        ax[1].plot(self.loss_list_D_fake, label='loss D fake', alpha=0.5)
        
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[0].set_xlabel('iterations')
        ax[1].set_xlabel('iterations')
        ax[0].set_ylabel('loss')
        ax[0].grid()
        ax[1].grid()
        ax[0].axhline(0)
        ax[1].axhline(0)
        
        savestring = str(self.results_dir / self.name / 'loss.png')
        fig.savefig(savestring)
        
        # stop


    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=8, trunc=1.0):
        self.GAN.eval()
        ext = "jpg" if not self.transparent else "png"
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(
            self.GAN.S, self.GAN.G, latents, n, trunc_psi=self.trunc_psi
        )
        save_image(
            generated_images,
            str(self.results_dir / self.name / f"{str(num)}.{ext}"),
            nrow=num_rows,
        )

        # moving averages

        generated_images = self.generate_truncated(
            self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi
        )
        save_image(
            generated_images,
            str(self.results_dir / self.name / f"{str(num)}-ema.{ext}"),
            nrow=num_rows,
        )

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]
        
        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)


    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi=0.75, num_image_tiles=8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(5000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images #.clamp_(0.0, 1.0)
    
    @torch.no_grad()
    def generate_truncated_fromW(self, G, w_styles, trunc_psi=0.75, num_image_tiles=8):
        latent_dim = G.latent_dim
        device = w_styles.device
        # if self.av is None:
        #     z = noise(2000, latent_dim)
        #     samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
        #     self.av = np.mean(samples, axis=0)
        #     self.av = np.expand_dims(self.av, axis=0)

        # w_space = []
        # for tensor, num_layers in style:
        #     tmp = S(tensor)
        #     av_torch = torch.from_numpy(self.av).cuda()
        #     tmp = trunc_psi * (tmp - av_torch) + av_torch
        #     w_space.append((tmp, num_layers))
        
        noise = torch.rand((self.batch_size, latent_dim), device='cuda')
        w_styles = w_styles.to('cuda')
        generated_images = evaluate_in_chunks_fromW(self.batch_size, G, w_styles, noise)
        return generated_images.clamp_(0.0, 1.0)
    
    def print_log(self, it):
        pl_mean = default(self.pl_mean, 0)
        print(
            f"it: {it} G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {pl_mean:.2f} | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}"
        )

    def model_name(self, num):
        return str(self.models_dir / self.name / f"model_{num}.pt")

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f"./models/{self.name}", True)
        rmtree(f"./results/{self.name}", True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {"GAN": self.GAN.state_dict()}

        if self.GAN.fp16:
            save_data["amp"] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [
                p for p in Path(self.models_dir / self.name).glob("model_*.pt")
            ]
            saved_nums = sorted(map(lambda x: int(x.stem.split("_")[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f"continuing from previous epoch - {name}")
        
        self.steps = int(name) * self.save_every

        load_data = torch.load(self.model_name(name))
        
        # print('------------ Summary Parameter StyleGAN2 ----------------')
        # print(
        #     "latent dim:", self.latent_dim,
        #     "\naug_prob:", self.aug_prob,
        #     "\ndataset_aug_prob:", self.dataset_aug_prob,
        #     "\naugmentation_types:", self.augmentation_types,
        #     "\ngradient_accumulate_every:", self.gradient_accumulate_every,
        #     "\nlearning rate:", self.lr,
        #     "\nlearning rate multiplier:", self.lr_mlp,
        #     "\ndiscriminator multiplier:", self.ttur_mult,
        #     "\nnoise:", self.noise,
        #     "\n")
        
        
        # make backwards compatible
        if "GAN" not in load_data:
            load_data = {"GAN": load_data}

        self.GAN.load_state_dict(load_data["GAN"])

        if self.GAN.fp16 and "amp" in load_data:
            amp.load_state_dict(load_data["amp"])
    
    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 1000, save_frames = False, latents=False):
        self.GAN.eval()
        # ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers
        
        # latents and noise
        
        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)
                    
        ratios = torch.linspace(0., 80, num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)
        
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    
    def generate_projection(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 1000, save_frames = False):
        self.GAN.eval()
        
        path = Path(str(self.results_dir / self.name / 'projection'))
        if not os.path.exists(path):
           path.mkdir(parents=True, exist_ok=True)
                            
        

        # projectiondir = 
        latentsdir = Path(str(self.results_dir / self.name / 'projection' / 'latents.th'))
        lossdir = Path(str(self.results_dir / self.name / 'projection' / 'losses.th'))
        imagedir = Path(str(self.results_dir / self.name / 'projection' / 'images.th'))
        recondir = Path(str(self.results_dir / self.name / 'projection' / 'recon.th'))
        
        if not os.path.exists(latentsdir):
            losses = []
            images = []
            latents = []
            reconstructed_images = []
            
            for i, image in tqdm(enumerate(self.valid_loader)):
                # image = next(self.valid_loader)
                # print(f'{i}-th image is projected')
                
                reconstructed_image, latent, loss = self.project(image,
                                                        batch_size=1,
                                                        lr=0.0005, 
                                                        max_epochs=3000,
                                                        epsilon=0.005,
                                                        plotting=False)
                
                losses.append(loss)
                images.append(image)
                latents.append(latent)
                reconstructed_images.append(reconstructed_image)
                
            images = torch.stack(images).squeeze(1).to('cpu')
            latents = torch.stack(latents).to('cpu')
            reconstructed_images = torch.stack(reconstructed_images).squeeze(1).to('cpu')
            losses = torch.stack(losses)
            
            torch.save(images, imagedir)
            torch.save(latents, latentsdir)
            torch.save(reconstructed_images, recondir)
            torch.save(losses, lossdir)
        
        else:
            images = torch.load(imagedir)
            latents = torch.load(latentsdir)
            reconstructed_images = torch.load(recondir)
            losses = torch.load(lossdir)
            
        subset_image_plus_recons = torch.cat( (images[:1*8], reconstructed_images[:1*8], 
                                               images[1*8:2*8], reconstructed_images[1*8:2*8],
                                               images[2*8:3*8], reconstructed_images[2*8:3*8],
                                               images[3*8:4*8], reconstructed_images[3*8:4*8],
                                               images[4*8:5*8], reconstructed_images[4*8:5*8])
                                             )

        print(f'Latent space reconstruction: {1-torch.mean(losses)} pm {torch.std(losses)}')
        

            
        fig, ax = plt.subplots(1,1)
        ax.axis("off")
        title=''
        fig.suptitle(title)
        padding=1
        grid = vutils.make_grid(subset_image_plus_recons[:num_image_tiles**2], padding=padding, normalize=False)
        ax.imshow(np.transpose(grid.cpu(),(1,2,0))[:,:,0], cmap='jet')
        
        for i in range(0, num_image_tiles+1, 2):
            ax.axhline((self.image_size + padding)*i, color='black')
            
        for i in range(0, num_image_tiles+1):
            ax.axvline((self.image_size + padding)*i, color='black')
            
        plt.show()
        save=True
        if save:
            fig.savefig(str(self.results_dir / self.name / f'{str(num)}.png'))
        plt.close(fig)
        
        self.generate_interpolation_for_wlatents(latents, num=num, num_image_tiles=num_image_tiles, trunc=trunc, 
                                   num_steps=num_steps, save_frames = save_frames)
        
        
    def generate_interpolation_for_wlatents(self, latents, num, num_image_tiles, trunc, 
                               num_steps, save_frames):
        self.GAN.eval()
        # ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers
        
        # calculate averages
        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, self.GAN.SE, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.std = np.std(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)
            self.std = np.expand_dims(self.std, axis=0)
        
        # how many stds are projections away from mean?
        # print(latents.size())
        # print(latents[0])
        # print(self.std)
        
        zscore = torch.subtract(latents, torch.tensor(self.av))/ torch.tensor(self.std)
        # print(zscore)

        # print(torch.mean(torch.abs(zscore)), torch.std(zscore))
        
        # w_space = []
        # for tensor, num_layers in style:
        #     tmp = S(tensor)
        #     av_torch = torch.from_numpy(self.av).cuda()
        #     tmp = trunc_psi * (tmp - av_torch) + av_torch
        #     w_space.append((tmp, num_layers))
        
        # latents and noise
        latents = latents[:num_rows**2, 0,:,:]
        device = latents.device
        
        # rng = torch.cat( [noise(num_rows ** 2, latent_dim).unsqueeze(1) for i in range(latents.size(1))], dim=1)
        # rng = rng.to(device)
        
        num_samples = latents.size(0)

        # Generate random indices along the first dimension
        random_indices1 = torch.randint(0, num_samples, (num_rows ** 2,))
        random_indices2 = torch.randint(0, num_samples, (num_rows ** 2,))
        
        latents_low = latents[random_indices1]
        latents_high = latents[random_indices2]
        n = image_noise(num_rows ** 2, image_size)
                    
        ratios = torch.linspace(0., 1, num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            # latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated_fromW(self.GAN.GE, interp_latents, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)
        
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    
    
    def give_average(self):
        # calculate averages
        latent_dim = self.latent_dim
        
        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, self.GAN.SE, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.std = np.std(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)
            self.std = np.expand_dims(self.std, axis=0)
        
        return torch.tensor(self.av)
    
    
    def project(self, 
                image,
                batch_size=1,
                lr=0.08, 
                max_epochs=5000,
                epsilon=0.0005,
                plotting=False,
                zlatent_space=False):
        
        self.GAN.eval()
        set_requires_grad(self.GAN.G, False)
        
        image = image.to('cuda')
        
        if zlatent_space:
            None
        else:
            # w_styles = torch.rand( (batch_size, self.GAN.G.num_layers, self.latent_dim), device='cuda')
            
            w_styles = self.give_average().unsqueeze(0)
            w_styles = w_styles.repeat((batch_size, self.GAN.GE.num_layers, 1))
            # print(w_styles.size())
            w_styles = w_styles.to('cuda')
            w_styles = torch.nn.Parameter(torch.autograd.Variable(w_styles))
        
        loss_list = []
            
        mae = torch.nn.L1Loss()
        
        optimizer = torch.optim.Adam([w_styles], lr=lr)
        # scheduler = ReduceLROnPlateau(optimizer, 
        #                               factor=0.9,
        #                               patience=5,
        #                               threshold=1e-4,
        #                               cooldown=5, 
        #                               verbose=True)

        scheduler = StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=- 1, verbose=False)
        
        for epoch in (range(max_epochs)):

            reconstructed_image = self.GAN.GE(w_styles, input_noise=None) # noise wird nicht berücksitigt
            loss = mae(reconstructed_image, image) # + beta*torch.abs(a) + beta*torch.abs(e)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            loss_list.append(loss.detach().cpu())
            
            if epoch%100 == 0 and plotting==True:
                fig, ax = plt.subplots(1,3, sharex='none', sharey='none')
                title = f'{epoch} ' + '{:.4f}'.format(loss.detach().cpu().numpy())
                fig.suptitle(title)
                ax[0].imshow(reconstructed_image.detach().cpu()[0,0,:,:])
                ax[1].imshow(image.detach().cpu()[0,0,:,:])
                ax[2].loglog((np.array(loss_list)))
                plt.show()
                
            if torch.allclose(image, reconstructed_image):
                print('Image reconstructed')
                break
            
            if loss < epsilon:
                break
        
        return reconstructed_image, w_styles.detach(), loss

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = Path(self.fid_dir + '/real')
        fake_path = Path(self.fid_dir + '/fake')

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists():
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                (real_batch, _, _) = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = 'png'

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim)
            noise = image_noise(self.batch_size, image_size)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))
        
        fid = fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], batch_size=16, 
                                                   device=noise.device, dims=2048)
        
        print(f"-------------- FID: {fid}--------------------")
        return 

def save_image(X, path, nrow=8, title='synthetic data'):
    plot_grid(batch=X, 
              batch_size=int(nrow**2),
              title=title, 
              savename=path, 
              normalize=False, 
              save=True)

def train(
    *,
    data="data",
    data_type="image_folder",
    results_dir="results",
    models_dir="models",
    name="default",
    new=True,
    load_from=-1,
    image_size=16,
    network_capacity=16,
    latent_dim=128,
    style_depth=8,
    fmap_max=256,
    transparent=False,
    batch_size=16,
    gradient_accumulate_every=1,
    num_train_steps=150000,
    learning_rate=2e-4,
    lr_mlp=0.1, # laut paper 0.01
    ttur_mult=1.5, #bei Mehdi auf 1.5, 2 laut lucedrian
    num_workers=1,
    save_every=1000,
    give_FID=False,
    generate=False,
    generate_interpolation=False,
    project=False,
    save_frames=False,
    num_image_tiles=8,
    trunc_psi=0.75,
    fp16=False,
    cl_reg=False,
    fq_dict_size=256,
    fq_layers=None,
    attn_layers=None,
    no_const=False,
    aug_prob=0.0, # differentiable augmentation
    dataset_aug_prob=0.0,
    mixed_prob=0.9,
    augmentation_types=None,
    noise=False,
    augment_surfaces_to_mean=False
):
    if fq_layers is None:
        fq_layers = []
    if attn_layers is None:
        attn_layers = []
    
    if CLUSTER:
        torch.cuda.set_device(hvd.local_rank())
        
    torch.backends.cudnn.benchmark = True
    model = Trainer(
        name,
        results_dir,
        models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        image_size=image_size,
        network_capacity=network_capacity,
        style_depth=style_depth,
        latent_dim=latent_dim,
        fmap_max=fmap_max,
        transparent=transparent,
        lr=learning_rate,
        lr_mlp=lr_mlp,
        ttur_mult=ttur_mult,
        num_workers=num_workers,
        save_every=save_every,
        trunc_psi=trunc_psi,
        fp16=fp16,
        cl_reg=cl_reg,
        fq_layers=fq_layers,
        fq_dict_size=fq_dict_size,
        attn_layers=attn_layers,
        no_const=no_const,
        aug_prob=aug_prob,
        dataset_aug_prob=dataset_aug_prob,
        mixed_prob=mixed_prob,
        augmentation_types=augmentation_types,
        noise=noise,
        augment_surfaces_to_mean=augment_surfaces_to_mean
    )

    if not new:
        model.load(load_from)
        if not CLUSTER:
            model.plot_loss()
        
    # else:
    #     model.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f"generated-{timestamp}"
        model.evaluate(samples_name, num_image_tiles)
        print(f"sample images generated at {results_dir}/{name}/{samples_name}")
        return

    if generate_interpolation:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f"generated-{timestamp}"
        model.generate_interpolation(
            samples_name, num_image_tiles, save_frames=save_frames
        )
        print(f"interpolation generated at {results_dir}/{name}/{samples_name}")
        return

    model.set_data_src(data)
    
    model.set_valid_data(data)
    
    
    if give_FID:
        model.calculate_fid(num_batches=75)
        return
    
    if project:
        
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f"projections-{timestamp}"
        model.generate_projection(
            samples_name, num_image_tiles, save_frames=save_frames
        )
        print(f"interpolation generated at {results_dir}/{name}/{samples_name}")
        return
    
        # image = next(iter(model.valid_loader))
        # model.project(image,
        #               batch_size=1,
        #               lr=0.0005, 
        #               max_epochs=3000,
        #               epsilon=0.005,
        #               plotting=True)
        return
        
    model.nbims = 0
    start = time.time()
    print(num_train_steps, model.steps)
    
    
    for it in range(num_train_steps - model.steps):
        model.train()
        if CLUSTER:
            if hvd.rank() == 0 and it % 100 == 0:
                model.print_log(it + model.steps)
                duration = time.time() - start
                print("nb images per second", model.nbims / duration)
        else:
            if it % 100 == 0:
                model.print_log(it + model.steps)
                duration = time.time() - start
                print("nb images per second", model.nbims / duration)
                
    duration = time.time() - start
    if CLUSTER:
        print(f"total images/sec: {(model.nbims/duration)*hvd.size()}")

    
    

    
    
    
    
    
if __name__ == "__main__":
    worldsize = 1
    if CLUSTER:
        hvd.init()
        worldsize = hvd.size()
        
    
    args = arg_parse()
    epochs = args.epochs
    dataset = get_dataset(args.data_path, is_train_or_valid='train')
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
    #                                       shuffle=True, num_workers=0)
    # for data in dataloader:
    #     plot_grid(data,64, title='', savename='', normalize=False, 
    #                   save=False)
    # stop
    
    steps_per_epoch = len(dataset) // (args.batch_size * worldsize)
    steps = steps_per_epoch * epochs
    
    print('number steps: ', steps)
    print(f'number training images: {steps * args.batch_size * worldsize}')
    
    augmentation_types = give_diff_augments()
    augmentation_types = ['translation', 'cutout']

    new = True
    load_from=-1
    if new==False:
        # args.name = 'styleGAN/05-25-2023_18-15-50'
        
        # args.name = 'styleGAN/05-26-2023_13-53-33'
        # load_from = r'31'

        # args.name = 'styleGAN/05-26-2023_14-45-43'
        # load_from = r'30'

        # args.name = 'styleGAN/05-30-2023_13-58-03'
        # load_from = r'16'
        
        # args.name = 'styleGAN/05-30-2023_14-03-51'
        # load_from = r'16'
        
        # args.name = 'styleGAN/05-25-2023_18-15-50'
        
        # args.name = 'styleGAN/05-31-2023_16-51-22' # zweit bestes modell FID: 67
        # load_from = r'70'
        
        # args.name = 'styleGAN/05-31-2023_16-52-36'   # bestes modell  FID: 60
        # load_from = r'139'
     
        # args.name = 'styleGAN/05-31-2023_16-54-59'     # FID: 84
        # load_from = r'63'    
        
        args.name = 'styleGAN/06-01-2023_15-16-29'      # FID: model 72: 45
                                                        #      model 62: 46     
                                                        #      model 52: 44
                                                        #      model 42: 47
                                                        #      model 32: 47
                                                        #      model 22: 52
                                                        #      model 12: 62
                                                        #      model 02: 224
        load_from = r'72'

        # args.name = 'styleGAN/06-01-2023_17-10-40'    #FID: 75
        # load_from = r'72'                               
        
        # args.name = 'styleGAN/06-05-2023_15-22-58'      # FID: 59
        # load_from = r'67'
        
        # args.name = 'styleGAN/06-05-2023_15-30-29'   # no convergence
        # load_from = r'63'
                
        # args.name = 'styleGAN/06-06-2023_17-17-56'   # 55
        # load_from = r'63'
        


    if CLUSTER and new==True:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        args.name = args.name+f'/{timestamp}/'

    train(
        network_capacity=24,
        latent_dim=32,
        style_depth=8,
        fmap_max=128,
        transparent=False,
        batch_size=args.batch_size,
        image_size=args.image_size,
        name=args.name,
        data=args.data_path,
        num_train_steps=steps,
        new=new,
        give_FID=False,
        generate=False,
        generate_interpolation=False,
        project=False,
        aug_prob=0.8,
        mixed_prob=0.9,
        save_every=1,
        load_from=load_from,
        augmentation_types=augmentation_types,
        trunc_psi=1,
        learning_rate=1e-4,
        lr_mlp=0.1, # laut paper 0.01
        ttur_mult=1.5,
        noise=False,
        augment_surfaces_to_mean=False,
        dataset_aug_prob=0
        # attn_layers=[1,2,3]
        # learning_rate=5e-4
    )
    
    
    # 1. versuch reconstruction accuracy: 0.9895042181015015 pm 0.0020329009275883436
    # styleGAN/05-25-2023_18-15-11: 0.9943723678588867 pm 0.0008958872058428824 # p=0.2
    # styleGAN/05-25-2023_18-15-50: 0.993811309337616 pm 0.001621982897631824 # p=0
    # styleGAN/05-26-2023_14-45-43: 0.9917133450508118 pm 0.0021088679786771536 # p=0.5, augmTypes=all

