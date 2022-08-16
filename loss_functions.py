#import tensorflow as tf
#import keras.backend as K
#from keras.losses import binary_crossentropy

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable, Function
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np

#beta = 0.25
#alpha = 0.25
#gamma = 2
#epsilon = 1e-5
#smooth = 1



class Semantic_loss_functions(nn.Module):
    def __init__(self, beta=0.25, alpha=0.25, gamma=2, epsilon=1e-5, smooth=1):
        #print("semantic loss functions initialized")
        super(Semantic_loss_functions, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + torch.finfo(torch.float32).eps) / (
                    torch.sum(y_true_f) + torch.sum(y_pred_f) + torch.finfo(torch.float32).eps)
        #将torch.epsilon()改为torch.finfo(torch.float32).eps

    def sensitivity(self, y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + torch.finfo(torch.float32).eps)

    def specificity(self, y_true, y_pred):
        true_negatives = torch.sum(
            torch.round(torch.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = torch.sum(torch.round(torch.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + torch.finfo(torch.float32).eps)

    def convert_to_logits(self, y_pred):
        y_pred = np.clip_by_value(y_pred, torch.finfo(torch.float32).eps,
                                  1 - torch.finfo(torch.float32).eps)
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_true, y_pred):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = self.beta / (1 - self.beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    #def focal_loss(self, y_true, y_pred):
     #   y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
      #                            1 - tf.keras.backend.epsilon())
     #   logits = tf.math.log(y_pred / (1 - y_pred))

      #  loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
        #                              alpha=self.alpha, gamma=self.gamma, y_pred=y_pred)

     #   return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss


    def bce_dice_loss(self, y_true, y_predict):
        Bce = nn.BCELoss()
        input = F.sigmoid(y_predict)
        target = y_true
        bce = Bce(input, target)
        dice= self.dice_loss(y_true, y_predict)
        loss = bce + dice
        return loss

    #def bce_dice_loss(self, y_true, y_pred):
        #loss = nn.BCELoss(F.sigmoid(y_pred), y_true) + self.dice_loss(y_true, y_pred)
        #return loss / 2.0

    def focal_loss(self, y_real, y_pred, eps=1e-8, gamma=2):
        # y_pred =  # hint: torch.clamp
        L = (y_pred.clamp(min=0) - y_pred * y_real + torch.log(1 + torch.exp(-torch.abs(y_pred)))).mean()
        focal_loss = 1 * (1 - torch.exp(-L)) ** gamma * L
        return focal_loss

    def bce_loss(self, y_pred, y_real):
        return (y_pred.clamp(min=0) - y_pred * y_real + torch.log(1 + torch.exp(-torch.abs(y_pred)))).mean()

    def bce_dice_focal_loss(self, y_true, y_predict, eps=1e-8, gamma=2):
        Bce = nn.BCELoss()
        input = F.sigmoid(y_predict)
        target = y_true
        bce = Bce(input, target)
        dice = self.dice_loss(y_true, y_predict)
        L = (y_predict.clamp(min=0) - y_predict * y_true + torch.log(1 + torch.exp(-torch.abs(y_predict)))).mean()
        focal_loss = -0.25 * (1 - torch.exp(-L)) ** gamma * L
        loss = bce + dice + focal_loss
        return loss

    def dice_focal_loss(self, y_true, y_predict, eps=1e-8, gamma=2):
        #Bce = nn.BCELoss()
        input = F.sigmoid(y_predict)
        target = y_true
        #bce = Bce(input, target)
        dice = self.dice_loss(y_true, y_predict)
        L = (y_predict.clamp(min=0) - y_predict * y_true + torch.log(1 + torch.exp(-torch.abs(y_predict)))).mean()
        focal_loss = -0.25 * (1 - torch.exp(-L)) ** gamma * L
        #loss = bce + dice + focal_loss
        loss = dice + focal_loss
        return loss

    def mse_dice_focal_loss(self, y_true, y_predict, eps=1e-8, gamma=2):
        #Bce = nn.BCELoss()
        MSE = nn.MSELoss()
        input = F.sigmoid(y_predict)
        target = y_true
        mse = MSE(input, target)
        dice = self.dice_loss(y_true, y_predict)
        L = (y_predict.clamp(min=0) - y_predict * y_true + torch.log(1 + torch.exp(-torch.abs(y_predict)))).mean()
        focal_loss = -0.25 * (1 - torch.exp(-L)) ** gamma * L
        loss = mse + (dice + focal_loss)/2
        #loss = mse + dice + focal_loss
        return loss

    def confusion(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = K.sum(y_pos * y_pred_pos)
        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + self.smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + self.smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

    def logcosh(self, true, pred):
        loss = torch.log(torch.cosh(pred - true))
        return torch.sum(loss)

    def huber(self, true, pred, delta=0.1):
        loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2), delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
        # return torch.mean(loss)
        return torch.sum(loss)

    def mae(self, true, pred):
        return torch.sum(torch.abs(true - pred))

    def smooth_l1_loss(self, input, target, beta=1. / 9):  # , size_average=True
        """
        very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        #if size_average:
            #return loss.mean()
        return loss.sum()
