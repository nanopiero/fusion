#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le 03/06/2024
@author: lepetit
# fonctions utiles pour la génération
# de données à fusionner
"""
from random import randint
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy 
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
class QPELoss_fcn(nn.Module):
    def __init__(self, cumuls_1h = False):
        super(QPELoss_fcn, self).__init__()
        self.regression_loss = nn.MSELoss()
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.cumuls_1h = cumuls_1h

    def forward(self, p, outputs, targets):

      # sans supervision imparfaite, avec spatialisation

      bs, nsteps, S, _ = targets.shape
      targets = targets.view(bs, 1, nsteps, S**2)
      mask = (targets>=0)
      masked_targets = targets[mask]
      outputs_rnr0 = outputs[:, :nsteps, ...].view(bs, 1, nsteps, S**2)
      outputs_rnr1 = outputs[:, nsteps:2*nsteps, ...].view(bs, 1, nsteps, S**2)
      masked_output_rnr = torch.cat([outputs_rnr0[mask].view(bs, 1, -1), outputs_rnr1[mask].view(bs, 1, -1)], dim=1)
      masked_target_rnr =  (masked_targets > 0).long().view(bs, -1)
      output_qpe = outputs[:, 2*nsteps:3*nsteps , ...]\
                                .view(bs, 1, nsteps, S**2)

      masked_output_qpe = output_qpe[mask][masked_targets > 0]
      masked_target_qpe = masked_targets[masked_targets > 0]

      loss_rnr = self.segmentation_loss(masked_output_rnr, masked_target_rnr)
      loss_qpe_1min = self.regression_loss(masked_output_qpe, masked_target_qpe)
      loss = 1/(2*p[0]**2) * loss_qpe_1min + 1/(2*p[1]**2) * loss_rnr


      if self.cumuls_1h:
        loss_qpe_1h = self.regression_loss(output_qpe.sum(dim=1)[mask[:,0,...]],
                                           targets.sum(dim=1)[mask[:,0,...]])
        loss += 1/(2*p[2]**2) * loss_qpe_1h
        loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)

      else:
        loss+= torch.log(1+p[0]**2+p[1]**2)

      with torch.no_grad():
        preds = masked_output_rnr.argmax(dim=1).flatten().cpu().numpy()
        targets = masked_target_rnr.flatten().cpu().numpy()
        # Compute the confusion matrix
        cm = confusion_matrix(targets, preds, labels=np.arange(2))


      if self.cumuls_1h:
        return loss_qpe_1min.item(), loss_qpe_1h.item(), loss_rnr.detach().item(), loss, cm
      else:
        return loss_qpe_1min.item(), loss_rnr.detach().item(), loss, cm

      
def compute_metrics(confusion_matrix):
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_ratio = fp / (tp + fp) if (tp + fp) > 0 else 0

    return accuracy, csi, sensitivity, specificity, false_alarm_ratio

