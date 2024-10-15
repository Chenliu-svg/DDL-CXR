import math
from typing import Any
import pandas as pd
import torch
import os
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import torchvision
from ldm.modules.diffusionmodules.openaimodel import QKVAttention, AttentionBlock
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score
# import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 4, patch_size: int = 4, emb_size: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x



class FusionTokens3inputAttnFuse(pl.LightningModule):
    """
    The prediciton model of DDL-CXR, using Attention as the final fusion.
    """
    def __init__(self,
                 task,
                 z1_embed_type='linear',
                 ehr_encoder_config=None,
                 cxr_transformer_config=None,
                 fusion_config=None,
                 vision_backbone='resnet34',
                 hidden_size=128,
                 hid_dim_1=128,
                 dropout=0.1,
                 conv_in_chan=4,
                 conv_out_chan=49,
                 z0_view_size=3136,
                 z0_view_size_chan1=784,
                 ehr_modal=False,
                 x0_modal=False,
                 z1_modal=False,
                 use_pos_weight=True,
                 fusion_way='na',
                 monitor='val/pr_auc',
                 mode='max',
                 max_epoch=100,
                 ckpt_path=None,

                 ignore_keys=[]
                 ):
        super().__init__()

       
        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=True)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break

        # self.cxr_feat_project=nn.Sequential(nn.Linear(d_visual, hidden_size),
        #                     nn.GELU(),nn.Dropout(dropout))
        self.cxr_feat_project=nn.Linear(d_visual, hidden_size)


        self.ehr_modal=ehr_modal
        self.z1_modal = z1_modal
        self.x0_modal=x0_modal


        self.mode = mode

        self.task = task
        self.max_epoch=max_epoch
        self.monitor=monitor
        
        
        if self.task=='mortality':
            pos_weight = torch.tensor([5.89])
            num_classes=1
            
        if self.task=='phenotype':
            if use_pos_weight:
                pos_weight = torch.tensor([1.66, 10.01, 10.45, 1.51, 3.09, 4.98, 3.22, 7.85, 2.07, 2.2, 6.96, 3.81, 1.51, 1.28, 0.92, 13.53, 3.55, 4.72, 6.16, 12.68, 8.3, 3.48, 1.95, 2.71, 3.44])
            else:
                pos_weight =torch.ones(25)
            num_classes=25


        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.num_classes = num_classes


        if self.ehr_modal:
            self.ehr_encoder=instantiate_from_config(ehr_encoder_config)


        self.z1_embed_type=z1_embed_type
        if self.z1_modal:
            # # encode image
            if z1_embed_type=='patchembedding':
               
                self.cxr_embdedding=PatchEmbedding()
                
                assert cxr_transformer_config is not None
                self.cxr_transformer=instantiate_from_config(cxr_transformer_config)

            elif z1_embed_type=='conv_trans':
                self.image_conv = nn.Conv2d(conv_in_chan, conv_out_chan, kernel_size=3, stride=1, padding=1)
                cxr_transformer_config['params']['feat_dim']=z0_view_size_chan1
                self.cxr_encoder=instantiate_from_config(cxr_transformer_config)

            elif self.z1_embed_type == 'conv_linear':
                self.image_conv = nn.Conv2d(conv_in_chan, conv_out_chan, kernel_size=3, stride=1, padding=1)
                self.image_fc=nn.Sequential(nn.Linear(z0_view_size_chan1, hidden_size),
                                nn.GELU())

            else:
                assert z1_embed_type=='linear'
                # (b,4,28,28) -> (b,3136) -> (b,3136) -> (b,128)
                self.cxr_linear=nn.Sequential(nn.Linear(z0_view_size, hidden_size),
                                nn.GELU())  #,nn.Dropout(dropout))


        self.fusion_way = fusion_way


        if fusion_way=='attention':
            # assert 'linear' not in cxr_embed_type, "linear cxr embedding cannot do attention"
            self.fusion_tf=instantiate_from_config(fusion_config)


        n_modal=int(ehr_modal+z1_modal+x0_modal)
        if fusion_way=='concat':
            self.concat_linear=nn.Sequential(nn.Linear(hidden_size*n_modal, hidden_size),
                            nn.GELU(),nn.Dropout(dropout))


        self.mlp_head  = nn.Sequential(nn.Linear(hidden_size, hid_dim_1),
                              nn.GELU(),
                            #   nn.Dropout(dropout),
                              nn.Linear(hid_dim_1,  self.num_classes)
                              )

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

    @torch.no_grad()
    def get_input(self, batch):
        # (ehr, y, x_0, sample_id,z1)
        ehr = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ehr = ehr.to(self.device)


        y= torch.stack(list(map(lambda x: x[1], batch)),dim=0)
        y = y.to(self.device).float()
       

        z = torch.stack(list(map(lambda x: x[4], batch)), dim=0)
        z = z.to(memory_format=torch.contiguous_format)
        z = z.to(self.device)

        x0 = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        x0 = x0.to(memory_format=torch.contiguous_format)
        x0 = x0.to(self.device)

        sample_id=torch.tensor(list(map(lambda x: x[3], batch)))
        sample_id = sample_id.to(self.device)

        return ehr, y, z, sample_id,x0





    def calculate_f1_score(self, predictions, target):

        preds = torch.argmax(predictions, dim=1).cpu()
        target = target.cpu()
        f1_micro = f1_score(target.numpy(), preds.numpy(), average='micro')
        f1_macro = f1_score(target.numpy(), preds.numpy(), average='macro')
        return torch.tensor(f1_micro).to(self.device), torch.tensor(f1_macro).to(self.device)

    def calculate_metrics(self, predictions, target):


        preds = predictions.numpy()
        target = target.numpy()

        roc_auc = roc_auc_score(target, preds)
        pr_auc = average_precision_score(target, preds)

        return torch.tensor(roc_auc).to(self.device), torch.tensor(pr_auc).to(self.device)



    def forward(self, ehr=None, z=None, x0=None ) -> Any:

        ehr_cls=None
        z1_cls=None
        x0_cls=None

        multimodal_reps = []

        if self.ehr_modal:
            # ehr_cls:(b,1,d_model)  encoded_ehr:(b,49,d_model)
            ehr_cls, encoded_ehr=self.ehr_encoder.encode(ehr)
            # ehr_cls:(b,1,d_model)
            encoded_ehr = encoded_ehr[:, 1:]
            multimodal_reps.append(encoded_ehr)

        if self.z1_modal:
            if self.z1_embed_type=='patchembedding':
                patched_z = self.cxr_embdedding(z)
                # z1_cls:(b,1,d_model)  encoded_cxr:(b,49,d_model)
                z1_cls, encoded_cxr = self.cxr_transformer.encode(patched_z)
                # z1_cls:(b,1,d_model)
                encoded_cxr = encoded_cxr[:, 1:]
                multimodal_reps.append(encoded_cxr)

            elif self.z1_embed_type == 'conv_trans':
                z=self.image_conv(z)
                b,out_chan,*spatial=z.shape
                #·Flatten the image tensor
                z=z.view(b, out_chan,-1)
                #cls=self.image_fc(z).mean(dim=1,keepdim=True)
                z1_cls, encoded_cxr=self.cxr_encoder.encode(z)
                z1_cls=z1_cls
                multimodal_reps.append(encoded_cxr)

            elif self.z1_embed_type == 'conv_linear':
                # (b,conv_out_channel,28,28)
                z=self.image_conv(z)
                b,out_chan,*spatial=z.shape
                #·Flatten the image tensor
                # (b,conv_out_channel, 28*28)
                z=z.view(b, out_chan,-1)
                # (b,49,128)
                z1_cls=self.image_fc(z).mean(dim=1,keepdim=True)
                # z1_cls, encoded_cxr=self.cxr_encoder.encode(z)
                # z1_cls=z1_cls.squeeze(dim=1)
                multimodal_reps.append(z1_cls)

            elif self.z1_embed_type=='linear':
                b = z.shape[0]
                z=z.view(b,-1)
                # z1_cls:(b,d_model)
                z1_cls=self.cxr_linear(z).unsqueeze(dim=1)
                multimodal_reps.append(z1_cls)
            else:
                raise NotImplementedError('unknown z1_emb_type')


        # encode cxr
        visual_feats = self.vision_backbone(x0)
        x0_cls = self.cxr_feat_project(visual_feats).unsqueeze(dim=1)
        multimodal_reps.append(x0_cls)

        seq = torch.cat(multimodal_reps, dim=1)

        if self.fusion_way == 'attention':
            fused_cls,_=self.fusion_tf.encode(seq)
            fused_cls=fused_cls.squeeze(dim=1)
       
        else:
            raise NotImplementedError('not implemented')

        output = self.mlp_head(fused_cls)
        if self.task == 'mortality':
            output = output.squeeze(1)

        return output


    def training_step(self, batch,batch_idx):

        ehr, y, z, sample_id,x0= self.get_input(batch)

        output = self(ehr=ehr, z=z, x0=x0)


        loss = self.loss(output, y)

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=y.shape[0])
       

        return loss

    def validation_step(self, batch, batch_idx):


        ehr, y, z, sample_id,x0= self.get_input(batch)

        output = self(ehr=ehr, z=z, x0=x0)


        loss = self.loss(output, y)

        preds = torch.sigmoid(output)
        

        return {'val_loss': loss, 'target': y, 'preds': preds}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs]).cpu()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu()

        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('val/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('val/pr_auc', pr_auc, prog_bar=True, on_epoch=True)

        
        del avg_loss,all_target,all_preds


    def test_step(self, batch, batch_idx) :



        ehr, y, z, sample_id,x0= self.get_input(batch)

        output = self(ehr=ehr, z=z, x0=x0)


        loss = self.loss(output, y)

        preds = torch.sigmoid(output)

        return {'test_loss': loss, 'target': y, 'preds': preds, 'sample_id':sample_id}

    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs]).cpu()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu()
        all_sample_id = torch.cat([x['sample_id'] for x in outputs]).cpu()


        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('test/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('test/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('test/pr_auc', pr_auc, prog_bar=True, on_epoch=True)

        
        if self.task=='mortality':
            data={'sample_id':all_sample_id,'target':all_target,'pred':all_preds}

            logdir=self.trainer.logger.save_dir
            df_results=pd.DataFrame(data=data)
            df_results.to_csv(os.path.join(logdir,'results.csv'))

            df_metrics=pd.DataFrame(data={'roc_auc':[roc_auc.item()],'pr_auc':[pr_auc.item()]})
            df_metrics.to_csv(os.path.join(logdir,'metrics.csv'))

        if self.task=='phenotype':


            namelist = ['sample_id']+["gt_" + str(i) for i in range(25)]+["gen_" + str(i) for i in range(25)]
            concatenated_tensor = np.concatenate((all_sample_id.unsqueeze(dim=1),all_target,all_preds), axis=1)

            df_results = pd.DataFrame(concatenated_tensor, columns=namelist)
            logdir=self.trainer.logger.save_dir

            df_results.to_csv(os.path.join(logdir,'results.csv'))





    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        schedular=CosineAnnealingLR(opt,T_max=self.max_epoch,eta_min=1e-7)
        return [opt],[schedular]


