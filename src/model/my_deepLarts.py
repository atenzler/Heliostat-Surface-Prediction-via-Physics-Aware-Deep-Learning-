"""
my_deepLarts.py

Main neural network definition for DeepLARTS, based on Lewen et al. (2024)
(https://doi.org/10.48550/arXiv.2408.10802), with slight modifications.

Includes:
- Utility functions for saving/loading models and folders
- Encoder building blocks (AdaIN and weight-demodulated variants)
- The StyleDeepLarts model (encoder + StyleGAN2 generator)
- Initialization function `init_deepLarts` for new or pretrained models
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from styleGAN2_surfaces import Trainer
import styleGAN2_surfaces
from functools import partial
from math import floor, log2
from PhysConUL_DownCont.Jan_NN_model import utils
from pathlib import Path
import os
from styleGAN2_surfaces import Conv2DMod
#from vit_pytorch.vit_flux_encoder import ViT_fusion
from utils import print_cluster, adjust_inputs


def init_folders(cfg, cluster, timestamp):
    # (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
    if cluster:
        import horovod

        modeldir = cfg.DIRECTORIES.JUWELS.MODELDIRF
        savemodeldir = Path(os.path.join(modeldir, timestamp))
        if horovod.torch.rank() == 0:
            savemodeldir.mkdir(parents=True, exist_ok=True)

    else:
        savemodeldir = Path(os.path.join('models', 'deepLarts', timestamp))
        savemodeldir.mkdir(parents=True, exist_ok=True)

    return savemodeldir


def save_statedic(deepLarts, optimizer, best=False, string=None):
    statedic = deepLarts.state_dict()
    optimizer = optimizer.state_dict()

    epoch = deepLarts.training_log["epoch"]
    state = {
        'statedic': statedic,
        'optimizer': optimizer,
        'architecture_args': deepLarts.architecture_args,
        'conv_enc_args': deepLarts.conv_enc_args,
        'trans_fuse_enc_args': deepLarts.trans_fuse_enc_args,
        'trans_flux_enc_args': deepLarts.trans_flux_enc_args,
        'styleGAN_args': deepLarts.styleGAN_args,
        'training_args': deepLarts.training_args,
        'data_args': deepLarts.data_args,
        'training_log': deepLarts.training_log
    }

    savemodeldir = deepLarts.savemodeldir
    if best == False:
        th.save(state, os.path.join(savemodeldir, f'deepLarts_{epoch}.pth'))
    else:
        th.save(state, os.path.join(savemodeldir, f'deepLarts_{string}.pth'))


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class DenseMappingToLatent(nn.Module):
    def __init__(self, ch, latent_dim, num_layers_generator, reduce_encoder_depth_by):
        super(DenseMappingToLatent, self).__init__()

        ninputs = int(ch * (2 ** (2 * (reduce_encoder_depth_by + 1))))
        noutputs = int(num_layers_generator * latent_dim)
        self.dense = nn.Linear(ninputs, noutputs)
        self.leaky_relu1 = leaky_relu()

    def forward(self, features):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        features = self.dense(features)
        return features


class styleMapping(nn.Module):
    def __init__(self, ninputs, chs, pdropout):
        super(styleMapping, self).__init__()
        # self.fc1 = nn.Linear(n_sun_position*n_embedding, neurons[0])
        # self.mapping_blocks = nn.ModuleList([nn.Linear(neurons[i], neurons[i+1]) for i in range(len(neurons)-1)])
        self.mean_style_blocks = nn.ModuleList([nn.Linear(ninputs, chs[i]) for i in range(len(chs))])
        self.std_style_blocks = nn.ModuleList([nn.Linear(ninputs, chs[i]) for i in range(len(chs))])
        self.dropout = nn.Dropout(pdropout)

    def forward(self, scalarinput):
        x = scalarinput.reshape(scalarinput.size(0), -1)

        styles_mean = []
        styles_std = []
        for i in range(len(self.mean_style_blocks)):
            style_mean = self.mean_style_blocks[i](x).unsqueeze(-1).unsqueeze(-1)
            style_mean = self.dropout(style_mean)
            style_std = self.std_style_blocks[i](x).unsqueeze(-1).unsqueeze(-1)
            style_std = self.dropout(style_std)
            styles_mean.append(style_mean)
            styles_std.append(style_std)

        return [styles_mean, styles_std]


# ------------------------ ENCODER -------------------------------------------
class encoderBlock_AdaIN(nn.Module):
    def __init__(self, in_ch, out_ch, residuals, use_scalar_input, epsilon_AdaIN, pdropout):
        super().__init__()
        self.residuals = residuals
        self.use_scalar_input = use_scalar_input
        self.epsilon_AdaIN = epsilon_AdaIN
        if self.residuals:
            self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
            self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.dropout = nn.Dropout(pdropout)
        self.leaky_relu = leaky_relu()

    def forward(self, features, style_mean1, style_std1, style_mean2, style_std2, lastBlockBool):

        # print(features.size(), style_mean1.size(), style_mean2.size(), style_std1.size() , style_std2.size())
        if self.use_scalar_input:
            if self.residuals:
                features = self.adain(features, style_mean1, style_std1)
                features = self.leaky_relu(self.conv1(features))
                features = self.dropout(features)
                features = self.adain(features, style_mean2, style_std2)
                features = self.leaky_relu(self.conv2(features) + features)
                features = self.down(features)
            else:
                features = self.adain(features, style_mean1, style_std1)
                features = self.leaky_relu(self.conv1(features))

        else:
            if self.residuals:
                features = self.leaky_relu(self.conv1(features))
                features = self.dropout(features)
                features = self.leaky_relu(self.conv2(features) + features)
                features = self.down(features)
            else:
                features = self.leaky_relu(self.conv1(features))

        return features

    def adain(self, content_features, style_mean, style_std):

        def calc_mean_std(features):
            batch_size, c = features.size()[:2]
            features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
            features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1)

            return features_mean, features_std

        content_mean, content_std = calc_mean_std(content_features)

        # avoid dividing by zero error
        content_std = content_std + self.epsilon_AdaIN

        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        normalized_features = normalized_features.float()
        return normalized_features


class encoderBlock_weight_demod(nn.Module):
    def __init__(self, in_ch, out_ch, residuals, use_scalar_input, pdropout):
        super().__init__()
        self.residuals = residuals
        if not use_scalar_input:
            raise Exception("use_scalar_input should be true for weigh demod!")

        if self.residuals:
            self.conv1 = Conv2DMod(in_ch, in_ch, 3)
            self.conv2 = Conv2DMod(in_ch, in_ch, 3)
            self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = Conv2DMod(in_ch, out_ch, 3)
            self.down = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(pdropout)
        self.leaky_relu = leaky_relu()

    def forward(self, features, style_mean1, style_std1, style_mean2, style_std2, lastBlockBool):
        style_mean1 = style_mean1.squeeze(-1).squeeze(-1)
        style_std1 = style_std1.squeeze(-1).squeeze(-1)
        if self.residuals:
            features = self.leaky_relu(self.conv1(features, style_mean1))
            features = self.dropout(features)
            features = self.leaky_relu(self.conv2(features, style_std1) + features)
            features = self.down(features)
        else:

            features = self.leaky_relu(self.conv1(features, style_mean1))
            features = self.dropout(features)
            features = self.down(features)

        return features


class Encoder(nn.Module):
    def __init__(self, chs, latent_dim, num_layers_generator,
                 reduce_encoder_depth_by, residuals, use_scalar_input,
                 dense_mapping_to_latent, epsilon_AdaIN, pdropout, use_weight_demod):
        super().__init__()
        self.residuals = residuals
        self.reduce_encoder_depth_by = reduce_encoder_depth_by
        self.use_scalar_input = use_scalar_input
        self.dense_mapping_to_latent = dense_mapping_to_latent
        self.epsilon_AdaIN = epsilon_AdaIN
        self.use_weight_demod = use_weight_demod

        if not self.use_weight_demod:
            self.enc_blocks = nn.ModuleList(
                [encoderBlock_AdaIN(chs[i], chs[i + 1], residuals, use_scalar_input, epsilon_AdaIN, pdropout) for i in
                 range(len(chs) - 1)])
        else:
            self.enc_blocks = nn.ModuleList(
                [encoderBlock_weight_demod(chs[i], chs[i + 1], residuals, use_scalar_input, pdropout) for i in
                 range(len(chs) - 1)])

        if not self.dense_mapping_to_latent:
            self.toLatent = nn.Conv2d(chs[-1], int(latent_dim * num_layers_generator / (
                        2 ** (2 * self.reduce_encoder_depth_by))),
                                      kernel_size=4, stride=2, padding=1)
        else:
            self.toLatent = DenseMappingToLatent(chs[-1], latent_dim, num_layers_generator,
                                                 self.reduce_encoder_depth_by)
        self.leaky_relu = leaky_relu()

    def forward(self, features, styles1, styles2):
        if self.use_scalar_input:
            styles_mean1, styles_std1 = styles1
            if self.residuals and not self.use_weight_demod:
                styles_mean2, styles_std2 = styles2

        lastBlock = len(self.enc_blocks)
        for i in range(len(self.enc_blocks)):

            block = self.enc_blocks[i]
            lastBlockBool = (lastBlock == i + 1)

            if self.use_scalar_input:
                if self.residuals and not self.use_weight_demod:
                    features = block(features, styles_mean1[i], styles_std1[i],
                                     styles_mean2[i], styles_std2[i], lastBlockBool)
                else:
                    features = block(features, styles_mean1[i], styles_std1[i],
                                     None, None, lastBlockBool)
            else:
                features = block(features, None, None, None, None, lastBlockBool)

        latent = self.leaky_relu(self.toLatent(features))
        return latent


# ----------------------------------------------- DEEP LARTS ---------------------------------------------
class styleDeepLarts(nn.Module):
    def __init__(self,
                 cfg,
                 architecture_args,
                 conv_enc_args,
                 trans_fuse_enc_args,
                 trans_flux_enc_args,
                 styleGAN_args,
                 data_args,
                 training_args=None,
                 device='cuda',
                 cluster=False
                 ):

        super(styleDeepLarts, self).__init__()

        self.cfg = cfg

        self.architecture_args = architecture_args
        self.conv_enc_args = conv_enc_args
        self.trans_fuse_enc_args = trans_fuse_enc_args
        self.trans_flux_enc_args = trans_flux_enc_args
        self.styleGAN_args = styleGAN_args
        self.training_args = training_args
        self.training_log = self.give_training_log()
        self.data_args = data_args

        self.flux_size = cfg.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE

        self.set_styleGAN2_trainer()

        self.min_alt = cfg.DEEPLARTS.TRAIN.SURFACE_MIN
        self.max_alt = cfg.DEEPLARTS.TRAIN.SURFACE_MAX

        self.generator = self.styleGAN2_trainer.GAN.GE
        self.latent_dim = self.styleGAN2_trainer.latent_dim
        self.num_layers_generator = self.styleGAN2_trainer.GAN.GE.num_layers
        self.num_layers_encoder = int(log2(self.flux_size) - 2 - conv_enc_args["reduce_encoder_depth_by"])
        self.num_init_filters = cfg.DEEPLARTS.TRAIN.NSUNPOS
        filters = [self.num_init_filters] + [(4 * conv_enc_args["network_capacity"]) * (2 ** i) for i in
                                             range(self.num_layers_encoder + 1)]

        set_fmap_max = partial(min, self.conv_enc_args["fmap_max"])
        self.enc_chs = list(map(set_fmap_max, filters))
        self.nsunpos = cfg.DEEPLARTS.TRAIN.NSUNPOS
        self.set_n_scalar_inputs()

        self.set_encoder()

        self.final_activation_string = self.architecture_args["final_activation"]
        self.final_activation = None
        self.set_final_activation(self.architecture_args["final_activation"])
        self.leaky_relu = leaky_relu()

        if cluster:
            self.savemodeldir = cfg.DIRECTORIES.JUWELS.MODELDIR
        else:
            self.savemodeldir = cfg.DIRECTORIES.LOCAL.MODELDIR

    def forward(self, flux, sunPos, helPos, targetID, canting_vecs=None):
        nsunPos = flux.size(1)
        if not nsunPos == sunPos.size(1):
            raise Exception("You must have as much flux densities as sun positions!")

        if not self.num_init_filters == nsunPos:
            flux, sunPos, targetID = self.adjust_inputs(flux, sunPos, targetID)

        batchsize = flux.size(0)
        device = flux.device

        if not self.architecture_args["use_transformer_encoder"]:
            if self.architecture_args["use_scalar_input"]:

                if not self.architecture_args["use_targetID"]:
                    scalar_list = [sunPos.view(batchsize, -1).to(device), helPos.view(batchsize, -1).to(device)]

                    # ⬅️ Append canting vectors if enabled
                    if self.architecture_args.get("use_canting_vecs", False) and canting_vecs is not None:
                        scalar_list.append(canting_vecs.view(batchsize, -1).to(device))

                    scalarinput = th.cat(scalar_list, dim=1)
                else:
                    scalar_list = [sunPos.view(batchsize, -1).to(device), targetID.view(batchsize, -1).to(device), helPos.view(batchsize, -1).to(device)]
                    if self.architecture_args.get("use_canting_vecs", False) and canting_vecs is not None:
                        scalar_list.append(canting_vecs.view(batchsize, -1).to(device))

                    scalarinput = th.cat(scalar_list, dim=1)

                stylesENC1 = self.style_mapping_encoder1(scalarinput)
                if self.conv_enc_args["residuals"] == True and not self.conv_enc_args["use_weight_demod"]:
                    stylesENC2 = self.style_mapping_encoder2(scalarinput)
                else:
                    stylesENC2 = None
            else:
                stylesENC1 = None
                stylesENC2 = None

            w_styles = self.encoder(flux, stylesENC1, stylesENC2)

        else:
            w_styles = self.encoder(flux, sunPos, helPos, targetID)
            w_styles = self.leaky_relu(w_styles)

        w_styles = w_styles.reshape((batchsize, self.num_layers_generator, self.latent_dim))

        noise = styleGAN2_surfaces.image_noise(1, 16)
        x = self.generator(w_styles, noise)

        if not self.final_activation == None:
            x = self.final_activation(x)

        x_recon = self.generator.to_physical_surface(x,
                                                     self.min_alt,
                                                     self.max_alt,
                                                     to_facets=True)

        #x_recon = utilsDL.zero_mean(x_recon)

        return x_recon, None

    def adjust_inputs(self, flux, sunPos, targetID):
        return adjust_inputs(flux, sunPos, targetID, self.num_init_filters)

    def set_final_activation(self, final_activation):
        if final_activation == 'sigmoid':
            self.final_activation = th.nn.Sigmoid()
        elif final_activation == 'leaky_relu':
            self.final_activation = leaky_relu()
        elif final_activation == None:
            self.final_activation = None
        else:
            raise Exception("Kein Gültiger Wert für die finale Aktivierung!")

    def set_n_scalar_inputs(self):
        ntargetIDs = 1 if self.architecture_args["use_targetID"] else 0
        if self.architecture_args["use_scalar_input"]:
            self.n_scalar_inputs = int(self.nsunpos * (3 + ntargetIDs) + 3)
            if self.architecture_args.get("use_canting_vecs", False):
                self.n_scalar_inputs += 24  # 4 facets × 6 values
        else:
            self.n_scalar_inputs = 0

    def set_styleGAN2_trainer(self):

        if self.styleGAN_args["new_styleGAN"] and self.styleGAN_args["freeze_styleGAN"]:
            raise Exception("Es macht keinen Sinn ein nicht trainiertes styleGAN zu freezen!")

        if not self.styleGAN_args["new_styleGAN"]:
            self.styleGAN2_trainer = Trainer(name="test",#self.styleGAN_args["styleGAN_name"],
                                             results_dir='results',
                                             models_dir=r'...\models',
                                             image_size=16,
                                             network_capacity=24)

            #self.styleGAN2_trainer.load(self.styleGAN_args["load_styleGAN_from"])
            self.styleGAN2_trainer.init_GAN()

        else:
            self.styleGAN_name = os.path.join("styleGAN", "untrained", "surface")
            self.load_styleGAN_from = 0
            self.styleGAN2_trainer = Trainer(name=self.styleGAN_name,
                                             results_dir='results',
                                             models_dir=r'...\models',
                                             image_size=16,
                                             network_capacity=24)

            self.styleGAN2_trainer.load(self.load_styleGAN_from)


        del self.styleGAN2_trainer.GAN.D
        del self.styleGAN2_trainer.GAN.G
        del self.styleGAN2_trainer.GAN.S
        del self.styleGAN2_trainer.GAN.SE

        if self.styleGAN_args["freeze_styleGAN"]:
            styleGAN2_surfaces.set_requires_grad(self.styleGAN2_trainer.GAN.GE, False)
        else:
            styleGAN2_surfaces.set_requires_grad(self.styleGAN2_trainer.GAN.GE, True)

    def set_encoder(self):

        if not self.architecture_args["use_transformer_encoder"]:
            self.encoder = Encoder(self.enc_chs,
                                   self.latent_dim,
                                   self.num_layers_generator,
                                   self.conv_enc_args["reduce_encoder_depth_by"],
                                   self.conv_enc_args["residuals"],
                                   self.architecture_args["use_scalar_input"],
                                   self.conv_enc_args["dense_mapping_to_latent"],
                                   self.conv_enc_args["pdropout_enc"],
                                   self.conv_enc_args["epsilon_AdaIN"],
                                   self.conv_enc_args["use_weight_demod"]
                                   )

            if self.architecture_args["use_scalar_input"]:

                self.style_mapping_encoder1 = styleMapping(ninputs=self.n_scalar_inputs,
                                                           chs=self.enc_chs,
                                                           pdropout=self.conv_enc_args["pdropout_styleMapping"])

                if self.conv_enc_args["residuals"] and not self.conv_enc_args["use_weight_demod"]:
                    self.style_mapping_encoder2 = styleMapping(self.n_scalar_inputs,
                                                               self.enc_chs,
                                                               self.conv_enc_args["pdropout_styleMapping"])
        else:
            self.encoder = ViT_fusion(dim=self.trans_fuse_enc_args["dim"],
                                      depth=self.trans_fuse_enc_args["depth"],
                                      image_size=self.flux_size,
                                      heads=self.trans_fuse_enc_args["heads"],
                                      mlp_dim=self.trans_fuse_enc_args["mlp_dim"],
                                      patch_size=self.trans_fuse_enc_args["patch_size"],
                                      channels=self.nsunpos,
                                      nstyleGAN_layers=self.num_layers_generator,
                                      styleLatent=self.latent_dim,
                                      dropout=self.trans_fuse_enc_args["dropout"],
                                      emb_dropout=self.trans_fuse_enc_args["emb_dropout"],
                                      flux_transformer_args=self.trans_flux_enc_args,
                                      targetID=self.architecture_args["use_targetID"]
                                      )
            # self.encoder = ViT_fusion(trans_flux_enc_args=self.trans_flux_enc_args,
            #                           trans_fuse_enc_args=self.trans_fuse_enc_args)

    def give_training_log(self):
        training_log = {"epoch": 0,
                        "loss_list_train": [],
                        "loss_list_valid": [],
                        "loss_list_test": [],
                        "loss_list_simreal_sim": [],
                        "loss_list_simreal_real": [],
                        "loss_list_valid_real": [],

                        "MPE_surface_train": [],
                        "MPE_surface_valid": [],
                        "MPE_surface_test": [],
                        "MPE_surface_simreal_sim": [],
                        "MPE_surface_simreal_real": [],

                        "MPE_surface_validReal": [],
                        "MPE_surface_validReal_sim": [],

                        "acc_flux_train": [],
                        "acc_flux_valid_sim": [],

                        "acc_flux_valid_real2unet": [],
                        "acc_flux_valid_real2defl": [],
                        "acc_flux_valid_sim2defl": [],
                        "acc_flux_valid_defl2unet": [],
                        "acc_flux_valid_ideal2unet": [],
                        "acc_flux_valid_ideal2defl": [],

                        "acc_flux_DLR_valid_real2unet": [],
                        "acc_flux_DLR_valid_real2defl": [],
                        "acc_flux_DLR_valid_sim2defl": [],
                        "acc_flux_DLR_valid_defl2unet": [],
                        "acc_flux_DLR_valid_defl2defl": []

                        }

        return training_log

    def set_finetuning_log(self):
        self.finetuning_log = {"epoch": 0,

                               "loss_list_valid_real": [],

                               "MPE_surface_trainReal": [],
                               "MPE_surface_trainReal_sim": [],

                               "acc_flux_train_real2unet": [],
                               "acc_flux_train_real2defl": [],
                               "acc_flux_train_sim2defl": [],
                               "acc_flux_train_defl2unet": [],
                               "acc_flux_train_ideal2unet": [],
                               "acc_flux_train_ideal2defl": [],

                               "acc_flux_DLR_train_real2unet": [],
                               "acc_flux_DLR_train_real2defl": [],
                               "acc_flux_DLR_train_sim2defl": [],
                               "acc_flux_DLR_train_defl2unet": [],
                               "acc_flux_DLR_train_defl2defl": [],

                               "MPE_surface_validReal": [],
                               "MPE_surface_validReal_sim": [],

                               "acc_flux_valid_real2unet": [],
                               "acc_flux_valid_real2defl": [],
                               "acc_flux_valid_sim2defl": [],
                               "acc_flux_valid_defl2unet": [],
                               "acc_flux_valid_ideal2unet": [],
                               "acc_flux_valid_ideal2defl": [],

                               "acc_flux_DLR_valid_real2unet": [],
                               "acc_flux_DLR_valid_real2defl": [],
                               "acc_flux_DLR_valid_sim2defl": [],
                               "acc_flux_DLR_valid_defl2unet": [],
                               "acc_flux_DLR_valid_defl2defl": []}
        return None

    def give_surface_loss(self, znctrlbatch, zcntrlbatch_pred):
        if self.training_args["training_loss"] == "L2":
            self = self.training_args["loss_multiplyier"] * F.mse_loss(zcntrlbatch_pred, znctrlbatch)
        elif self.training_args["training_loss"] == "MAE":
            loss_surface = 1000 * utils.give_mean_pixel_error(znctrlbatch, zcntrlbatch_pred)
        elif self.training_args["training_loss"] == "ACC":
            loss_surface = 1 - utils.give_accuracy(znctrlbatch, zcntrlbatch_pred)
        else:
            raise Exception("The choosen training loss does not exist.")

        return loss_surface

    def give_flux_loss(self, fluxbatch, fluxbatch_prediction):
        if self.training_args["train_on_flux_density"]:

            # print(fluxbatch_[:,randint,:,:].size(), fluxbatch_prediction.size())
            acc_flux = utils.give_accuracy(fluxbatch,
                                           fluxbatch_prediction)
            loss_flux = 1 - acc_flux
        else:
            loss_flux = 0

        return loss_flux

    def print_training_state(self, duration, rank, lr):
        loss_multiplyier = self.training_args["loss_multiplyier"]
        if (rank == 0 or rank == ''):
            if (self.training_log["epoch"] <= self.training_args["give_flux_loss_every"]):
                print(
                    'Epoch [%d / %d] \n l2_train: %.2f, l2_valid: %.2f, MPE_train: %.2f, MPE_valid: %.2f \n MPE_validReal: %.2f, MPE_validReal_sim: %.2f,  lr: %.5f, t: %d s' % (
                    self.training_log["epoch"],
                    self.training_args["nepochs"],
                    loss_multiplyier * self.training_log["loss_list_train"][-1],
                    loss_multiplyier * self.training_log["loss_list_valid"][-1],
                    self.training_log["MPE_surface_train"][-1],
                    self.training_log["MPE_surface_valid"][-1],
                    self.training_log["MPE_surface_validReal"][-1],
                    self.training_log["MPE_surface_validReal_sim"][-1],
                    lr,
                    duration.total_seconds()))

            else:
                print(
                    'Epoch [%d / %d] \n l2_train: %.2f, l2_valid: %.2f, MPE_train: %.2f, MPE_valid: %.2f \n MPE_validReal: %.2f, MPE_validReal_sim: %.2f,  ACC_real: %.2f, ACC_sim: %.2f, lr: %.5f, t: %d s' % (
                    self.training_log["epoch"],
                    self.training_args["nepochs"],
                    loss_multiplyier * self.training_log["loss_list_train"][-1],
                    loss_multiplyier * self.training_log["loss_list_valid"][-1],
                    self.training_log["MPE_surface_train"][-1],
                    self.training_log["MPE_surface_valid"][-1],
                    self.training_log["MPE_surface_validReal"][-1],
                    self.training_log["MPE_surface_validReal_sim"][-1],
                    self.training_log["acc_flux_valid_real2unet"][-1],
                    self.training_log["acc_flux_valid_sim2defl"][-1],
                    lr,
                    duration.total_seconds()))


def give_load_statedicdir(modeldir, name_deepLarts, load_from_deepLarts):
    return os.path.join(modeldir, name_deepLarts, f'deepLarts_{load_from_deepLarts}.pth')


def init_deepLarts(cfg,
                   new_deepLarts,
                   name_deepLarts,
                   load_from_deepLarts,
                   data_args,
                   architecture_args=None,
                   convolution_encoder_args=None,
                   transformer_fusion_encoder_args=None,
                   transformer_flux_encoder_args=None,
                   styleGAN_args=None,
                   training_args=None,
                   timestamp='no_timestemp',
                   device='cuda',
                   cluster=True,
                   rank=''):
    if new_deepLarts:

        if architecture_args["use_targetID"]:
            if not architecture_args["use_scalar_input"]:
                raise Exception("When you use a targetID you must enable scalar inputs")

        print_cluster(string="new deepLarts is inititalized!",
                      cluster=cluster,
                      rank=rank)

        savemodeldir = init_folders(cfg, cluster, timestamp)

        deepLarts = styleDeepLarts(cfg,
                                   architecture_args=architecture_args,
                                   conv_enc_args=convolution_encoder_args,
                                   trans_fuse_enc_args=transformer_fusion_encoder_args,
                                   trans_flux_enc_args=transformer_flux_encoder_args,
                                   styleGAN_args=styleGAN_args,
                                   training_args=training_args,
                                   data_args=data_args,
                                   device=device,
                                   cluster=cluster
                                   )

        deepLarts.savemodeldir = savemodeldir
        deepLarts.name_deepLarts = timestamp
        if rank == '' or rank == 0:
            utils.print_model_summary(deepLarts)

    else:
        modeldir = cfg.DIRECTORIES.JUWELS.MODELDIR if cluster else cfg.DIRECTORIES.LOCAL.MODELDIR
        loaddir = give_load_statedicdir(modeldir, name_deepLarts, load_from_deepLarts)
        config = th.load(loaddir, map_location=device)

        try:
            data_args = config["data_args"]
        except:
            data_args = {"appl_low_passfilter_on_data": False}
            print("Attention. data_args were not found!")

        deepLarts = styleDeepLarts(cfg=cfg,
                                   architecture_args=config["architecture_args"],
                                   conv_enc_args=config["conv_enc_args"],
                                   trans_fuse_enc_args=config["trans_fuse_enc_args"],
                                   trans_flux_enc_args=config["trans_flux_enc_args"],
                                   styleGAN_args=config["styleGAN_args"],
                                   training_args=config["training_args"],
                                   data_args=data_args,
                                   device=device,
                                   cluster=cluster
                                   )

        deepLarts.load_state_dict(config["statedic"])
        deepLarts.training_log = config["training_log"]
        deepLarts.load_from_deepLarts = load_from_deepLarts
        deepLarts.name_deepLarts = name_deepLarts
        if cluster:
            savemodeldir = cfg.DIRECTORIES.JUWELS.MODELDIR
        else:
            savemodeldir = cfg.DIRECTORIES.LOCAL.MODELDIR

        deepLarts.savemodeldir = os.path.join(savemodeldir, name_deepLarts)
        deepLarts.versiondir = os.path.join(savemodeldir, name_deepLarts, 'results_' + load_from_deepLarts)

        if rank == '' or rank == 0:
            utils.print_model_summary(deepLarts)

    return deepLarts













