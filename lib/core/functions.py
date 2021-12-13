import os
import time
import logging
import numpy as np

import torch
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def train(train_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir, config):
    model.train()
    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0
    loss_pred_dis = 0.0

    for batch_idx, data in enumerate(train_loader):
        img, label, mask_label, imgPaths = data

        img, label = img.cuda(), label.cuda()
        mask_label = mask_label.cuda()
        
        features = model(img)

        # compute output
        if config.TRAIN.MODE == 'Clean' or config.TRAIN.MODE == 'Occ':
            feature = features[-1]
            output = classifier(feature, label)
            loss = criterion(output, label)

        elif config.TRAIN.MODE == 'Mask':
            output, loss, loss_cls, loss_pred, mask, preds = occ_train(features, label, mask_label, config, classifier, criterion)
        else:
            raise ValueError('Unknown training mode!')

        loss_display += loss.item()
        if config.TRAIN.MODE == 'Mask':
            loss_cls_dis += loss_cls.item()
            loss_pred_dis += loss_pred.item()
        else:
            loss_cls_dis = 0
            loss_pred_dis = 0
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:
            
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq
            loss_cls_dis /= num_freq
            loss_pred_dis /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    iters, loss_display, acc, time_used, speed) + INFO)
            if config.TRAIN.MODE == 'Mask':
                logger.info('Cls Loss: {:.4f}; Pred Loss: {:.4f}*{}'.format(loss_cls_dis, loss_pred_dis, config.LOSS.WEIGHT_PRED))
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)
                if config.TRAIN.MODE == 'Mask':
                    sw.add_scalar('CLS_LOSS', loss_cls_dis, iters)
                    sw.add_scalar('PRED_LOSS', loss_pred_dis, iters)
            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0
            loss_pred_dis = 0.0 

def occ_train(features, label, mask_label, config, classifier, criterion):
    fc_mask, mask, vec, fc = features

    output = classifier(fc_mask, label)
    loss_cls = criterion(output, label)

    loss_pred = criterion(vec, mask_label)
    preds = vec.cpu().detach().numpy()
    preds = np.argmax(preds, axis=1)

    loss = loss_cls + config.LOSS.WEIGHT_PRED * loss_pred

    return output, loss, loss_cls, loss_pred, mask, preds

