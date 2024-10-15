import os
from typing import Optional
from PIL import Image
import h5py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.optim.lr_scheduler import CosineAnnealingLR
from ldm.util import instantiate_from_config
import torchvision.models as models

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff



    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x


    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict



    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # @torch.no_grad()
    # def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
    #     log = dict()
    #     z1 = batch[0]
    #     z1 = z1.to(self.device)
    #     stored_z1=self.decode(z1)
    #     log['stored_z1']=stored_z1
    #
    #     x0=batch[1]
    #     log["x0"] = x0
    #
    #     x1 = batch[2]
    #     log["x1"] = x1
    #
    #     return log
#     def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         if only_inputs:
#             log["inputs"] = x
#             return log
#         xrec, _ = self(x)
#         if x.shape[1] > 3:
#             # colorize with random projection
#             assert xrec.shape[1] > 3
#             x = self.to_rgb(x)
#             xrec = self.to_rgb(xrec)
#         log["inputs"] = x
#         log["reconstructions"] = xrec
#         if plot_ema:
#             with self.ema_scope():
#                 xrec_ema, _ = self(x)
#                 if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
#                 log["reconstructions_ema"] = xrec_ema
#         return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,  
                 mode='min',
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 
                 ):
        super().__init__()
        self.mode=mode

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        


    def init_from_ckpt(self, path, ignore_keys=list()):

        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        # return sd

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    @torch.no_grad()
    def get_input(self, batch):
        x =  torch.stack(batch, dim=0)

        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        x = x.to(memory_format=torch.contiguous_format).float()
        del batch
        return x

    @torch.no_grad()
    def get_input_for_offline(self, batch, bs=None):
        sample_id = torch.tensor(list(map(lambda x: x[0], batch)))

        sample_id = sample_id.to(self.device)

        # x0
        x0 = torch.stack(list(map(lambda x: x[1], batch)),dim=0)

        x0 = x0.to(memory_format=torch.contiguous_format).float()
        x0 = x0.to(self.device)

        # encoder_posterior is a DiagonalGaussianDistribution
        encoder_posterior = self.encode(x0)
        # z0: latent vector sampled from the DiagonalGaussianDistribution
        z0 = encoder_posterior.sample().detach()

        out = [sample_id, z0]

        return out
    



    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

          
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9),weight_decay=1e-4)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def get_last_encoder_layer(self):
        return self.encoder.conv_out.weight
    

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        # get ground truth
        x = self.get_input(batch)
        x = x.to(self.device)
        if not only_inputs:
            # in forward function: xrec is sampled from posterior output from the encoder
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
    
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            # get reconstruction
            log["reconstructions"] = xrec
        log["inputs"] = x
    
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class AutoencoderKLLabel(AutoencoderKL):
    """
    VAE of DDL-CXR, adding chexpert task to AutoencoderKL
    """

    def __init__(self, ddconfig, lossconfig, embed_dim, gen_weight=10000, disc_weight=1,rec_weight=1, z0_view_size=3136, num_classes=14, mode='min', ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None):
        super().__init__(ddconfig, lossconfig, embed_dim, mode, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor)

        # self.pos_weight=torch.tensor([2.48,2.65,11.10,3.26,16.55,41.05,49.41,1.95,6.03,2.02,141.56,10.25,15.56,0.63])
        self.label_loss=nn.BCELoss()
        self.label_head=nn.Sequential(nn.Linear(z0_view_size,  num_classes),nn.Sigmoid())
        self.gen_weight=gen_weight
        self.disc_weight=disc_weight
        self.rec_weight=rec_weight

        self.CLASSES=['Atelectasis',
                    'Cardiomegaly',
                    'Consolidation',
                    'Edema',
                    'Enlarged Cardiomediastinum',
                    'Fracture',
                    'Lung Lesion',
                    'Lung Opacity',
                    'No Finding',
                    'Pleural Effusion',
                    'Pleural Other',
                    'Pneumonia',
                    'Pneumothorax',
                    'Support Devices']

    def calculate_adaptive_label_weight(self, nll_loss, label_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            l_grads = torch.autograd.grad(label_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            l_grads = torch.autograd.grad(label_loss, self.last_layer[0], retain_graph=True)[0]

        l_weight = torch.norm(nll_grads) / (torch.norm(l_grads) + 1e-4)
        l_weight = torch.clamp(l_weight, 0.0, 1e4).detach()
        
        return l_weight    

    @torch.no_grad()
    def get_input(self, batch):
        x = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        x = x.to(memory_format=torch.contiguous_format).float()
        x=x.to(self.device)

        label = torch.stack(list(map(lambda x: x[1], batch)), dim=0).to(torch.float)
        label=label.to(self.device)
        del batch
        return x,label

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        gt_x,gt_label = self.get_input(batch)
        reconstructions, posterior, z = self(gt_x)

        b = z.shape[0]
        z=z.view(b,-1)
        # cxr_cls:(b,d_model)
        gen_label=self.label_head(z)
        label_loss=self.label_loss(gen_label,gt_label)
        self.log("train/label_loss", label_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            

        if optimizer_idx == 0:
            aeloss, log_dict_ae, nll_loss = self.loss(gt_x, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

           
        
            label_loss_weight=self.calculate_adaptive_label_weight(nll_loss,label_loss,last_layer=self.get_last_encoder_layer())
            # label_loss_weight=10000
            gen_loss=aeloss+label_loss_weight*label_loss
            self.log("train/select_loss", gen_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/label_loss_weight", label_loss_weight, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            # return final_loss
            return {'loss': gen_loss, 'target': gt_label.detach(), 'preds': gen_label.detach()}
    

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gt_x, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            return discloss

          
    
    def training_epoch_end(self, outputs) -> None:
        print('call training epoch end')
        target = torch.cat([x['target'] for x in outputs[0]])
        predictions = torch.cat([x['preds'] for x in outputs[0]])

        preds = predictions.cpu().numpy()
        target = target.cpu().numpy()

        mean_roc=0
        mean_pr=0
        class_num=target.shape[1]
        for i in range(class_num):
            # print(f'class:{i}')
            roc_auc = roc_auc_score(target[:,i], preds[:,i])
            mean_roc+=roc_auc
            roc_auc=torch.tensor(roc_auc).to(self.device)
            
            pr_auc = average_precision_score(target[:,i], preds[:,i])
            mean_pr+=pr_auc
            pr_auc=torch.tensor(pr_auc).to(self.device)
             
        # mean auc
        self.log_dict( {'train/train_mean_roc':torch.tensor(mean_roc/class_num).to(self.device),'train/train_mean_pr':torch.tensor(mean_pr/class_num).to(self.device)}, prog_bar=True, on_epoch=True)
        del target,predictions,preds
        

            

    def validation_step(self, batch, batch_idx):
        gt_x,gt_label = self.get_input(batch)
        reconstructions, posterior, z = self(gt_x)

        b = z.shape[0]
        z=z.view(b,-1)
        # cxr_cls:(b,d_model)
        gen_label=self.label_head(z)
        label_loss=self.label_loss(gen_label,gt_label)

        self.log("val/label_loss", label_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
        aeloss, log_dict_ae, nll_loss = self.loss(gt_x, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(gt_x, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss=log_dict_ae["val/rec_loss"]
        
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return {'target': gt_label.detach(), 'preds': gen_label.detach()}
        
    

    def validation_epoch_end(self, outputs) -> None:
        print('call validation epoch end')
        target = torch.cat([x['target'] for x in outputs],dim=0)
        predictions = torch.cat([x['preds'] for x in outputs],dim=0)

        preds = predictions.cpu().numpy()
        target = target.cpu().numpy()

        mean_roc=0
        mean_pr=0
        class_num=target.shape[1]
        for i in range(class_num):
            # print(f'class:{i}')
            roc_auc = roc_auc_score(target[:,i], preds[:,i])
            mean_roc+=roc_auc
            roc_auc=torch.tensor(roc_auc).to(self.device)
            
            pr_auc = average_precision_score(target[:,i], preds[:,i])
            mean_pr+=pr_auc
            pr_auc=torch.tensor(pr_auc).to(self.device)
                 
        # mean auc
        self.log_dict({'val_mean_roc':torch.tensor(mean_roc/class_num).to(self.device),'val_mean_pr':torch.tensor(mean_pr/class_num).to(self.device)}, prog_bar=True, on_epoch=True)
        del target,predictions,preds
        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                list(self.label_head.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
