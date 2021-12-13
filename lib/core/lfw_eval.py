from PIL import Image
import numpy as np
import copy
import os

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import logging
from tensorboardX import SummaryWriter

cudnn.benchmark = True

logger = logging.getLogger(__name__)

def extractDeepFeature(img, model, is_gray, mask=None, binary=False):
    img = img.to('cuda')
    fc_mask, mask, vec, fc = model(img, mask)
    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    
    return fc, fc_mask, mask


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def cosine_similarity(f1, f2):
    # compute cosine_similarity for 2-D array
    f1 = f1.numpy()
    f2 = f2.numpy()

    A = np.sum(f1*f2, axis=1)
    B = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-5

    return A / B

def binary_mask(mask, thres=0.3):
    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    mask = torch.where(mask > thres, ones, zeros)
    return mask

def soft_binary_mask(mask, thres=0.3):
    zeros = torch.zeros_like(mask)
    mask = torch.where(mask > thres, mask, zeros)
    # expand to [0, 1]
    # mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
    return mask

def mask_ratio(mask, ratio=0.2):
    index = int(ratio * mask.nelement())
    thres = torch.sort(mask.flatten())[0][index].item()
    return binary_mask(mask, thres)

def cal_mask_stat(mask, num=11):
    grid = np.linspace(0, 1, num)
    mask_stats = np.zeros((num,))

    for i in range(num-1):
        low = grid[i]
        high = grid[i+1]
        A = mask >= low
        B = mask < high
        count = np.sum(A * B)
        mask_stats[i] = count
    mask_stats[num-1] = mask.size - np.sum(mask_stats)
    return mask_stats

def compute_distance(img1, img2, model, mode, indicator, flag, is_gray, binary_thres, soft_binary):
        f2, f2_mask, mask2 = extractDeepFeature(img2, model, is_gray, None)
        f1, f1_mask, mask1 = extractDeepFeature(img1, model, is_gray, None)

        mask_occ = mask2.cpu().detach().numpy()
        mask_clean = mask1.cpu().detach().numpy()

        # if soft_binary:
            # mask1 = soft_binary_mask(mask1, binary_thres)
            # mask2 = soft_binary_mask(mask2, binary_thres)
        # else:
            # mask1 = binary_mask(mask1, binary_thres)
            # mask2 = binary_mask(mask2, binary_thres)

        f2, f2_mask, mask2 = extractDeepFeature(img2, model, is_gray, mask2)
        f1, f1_mask, mask1 = extractDeepFeature(img1, model, is_gray, mask1)

        if mode != 'Mask':
            f1_mask, f2_mask = f1, f2

        if indicator == 'Baseline':
            distance = cosine_similarity(f1, f2)
        else:
            distance = cosine_similarity(f1_mask, f2_mask)

        flag = flag.squeeze().numpy()
        return np.stack((distance, flag), axis=1), mask_clean, mask_occ

def tar_far(far, predicts, name):
    posi_scores = []
    nega_scores = []
    for pairs in predicts:
        if pairs[1] == 0:
            nega_scores.append(pairs[0])
        else:
            posi_scores.append(pairs[0])
    posi_scores.sort(reverse=True)
    nega_scores.sort(reverse=True)

    thres_idx = int(len(nega_scores) * far)
    threshold = nega_scores[thres_idx]

    num_correct = np.array(np.array(posi_scores) >= threshold, dtype=int).sum()
    tar = float(num_correct) / len(posi_scores)
    logger.info('{}: far {} leads to tar {:.4f}'.format(name, far, tar))
    return tar

def obtain_acc(predicts, num_class, name, start):
    accuracy = []
    thd = []
    folds = KFold(n=num_class, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    end = time.time()
    time_used = (end - start) / 60.0
    logger.info('{}_LFW_ACC={:.4f} std={:.4f} thd={:.4f} time_used={:.4f} mins'.format(name, np.mean(accuracy), np.std(accuracy), np.mean(thd), time_used))
    return np.mean(accuracy)


def eval(model, model_path, config, test_loader, tb_log_dir, epoch, is_gray=False):
    indicator = 'FROM'
    if model_path:
        if 'baseline' in model_path or '_occ_' in model_path:
            indicator = 'Baseline'
        logger.info('using model:{}, indicator:{}'.format(model_path, indicator))
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

    predicts = np.zeros(shape=(len(test_loader.dataset), 2))
    predicts_occ = np.zeros(shape=(len(test_loader.dataset), 2))

    model.eval()
    start = time.time()

    cur = 0
    with torch.no_grad():
        for batch_idx, (img1, img2, img2_occ, flag) in enumerate(test_loader):
            predicts[cur:cur+flag.shape[0]], _, _ = compute_distance(img1, img2, model, config.TEST.MODE, indicator, flag, is_gray, config.TRAIN.BINARY_THRES, config.TRAIN.SOFT_BINARY)
            predicts_occ[cur:cur+flag.shape[0]], _, _ = compute_distance(img1, img2_occ, model, config.TEST.MODE, indicator, flag, is_gray, config.TRAIN.BINARY_THRES, config.TRAIN.SOFT_BINARY)
            cur += flag.shape[0]
    assert cur == predicts.shape[0]

    accuracy = obtain_acc(predicts, test_loader.dataset.num_pairs, 'Clean', start)
    accuracy_occ = obtain_acc(predicts_occ, test_loader.dataset.num_pairs, 'Occ', start)

    tar_occ = tar_far(1e-4, predicts_occ, 'Occ')

    # visualize the masks stats
    writer = SummaryWriter(tb_log_dir)
    writer.add_scalar('CLEAN_LFW_ACC', np.mean(accuracy), epoch)
    writer.add_scalar('OCC_LFW_ACC', np.mean(accuracy_occ), epoch)
    writer.close()

    return np.mean(accuracy), np.mean(accuracy_occ), tar_occ

