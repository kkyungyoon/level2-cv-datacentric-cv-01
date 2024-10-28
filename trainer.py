import torch
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from model import EAST
from torch.optim import lr_scheduler



class OCRTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = EAST()
        self.learning_rate = args.learning_rate
        self.milestones = [args.epoch // 2]
        self.gamma = 0.1


    def forward(self, x):
        output = self.backbone(x)
        return output 
    
    # def on_train_epoch_start(self):
    #     self.backbone.train()

    def training_step(self, batch):

        img, gt_score_map, gt_geo_map, roi_mask = batch

        loss, extra_info = self.backbone.train_step(img, gt_score_map, gt_geo_map, roi_mask)

        # current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # self.log("learning_rate", current_lr, on_step=True, on_epoch=False)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        self.log("Cls loss",extra_info['cls_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Angle loss",extra_info['angle_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("IoU loss",extra_info['iou_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # def on_validation_epoch_start(self):
    #     self.backbone.eval()


    def validation_step(self, batch):
        img, gt_score_map, gt_geo_map, roi_mask = batch

        loss, extra_info = self.backbone.train_step(img, gt_score_map, gt_geo_map, roi_mask)

        self.log("val_loss", loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        self.log("val_Cls loss",extra_info['cls_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log("val_Angle loss",extra_info['angle_loss'] , prog_bar=True, logger=True)
        self.log("val_IoU loss",extra_info['iou_loss'], prog_bar=True, logger=True)


    # def predict_step(self, batch, batch_idx):
    #     if self.model_type == 'swin':
    #         x, _, _ = self(batch)
    #     else:
    #         x = self(batch)
    #     logits = F.softmax(x, dim=1)
    #     preds = logits.argmax(dim=1)

    #     if self.k_fold_option:
    #         return preds, logits
    #     else:
    #         return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
