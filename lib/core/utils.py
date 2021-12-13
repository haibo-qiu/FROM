import os
import time
import torch
import torchvision
import random
import socket
import logging
import numpy as np
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from lib.core.config import get_model_name

logger = logging.getLogger(__name__)

def mp_print(msg):
    if dist.get_rank() == 0:
        print(msg)

# ############## logger related ######################
def create_temp_logger():
    output_dir = 'temp'
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir

    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = final_output_dir / log_file

    fmt = "[%(asctime)s] %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # add file handler to save the log to file
    fh = logging.FileHandler(final_log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    tensorboard_log_dir = root_output_dir / 'tensorboard'
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    model += '-' + cfg.TRAIN.MODE
    cfg_name = os.path.basename(cfg_name).split('.')[0] 

    pretrained = 1 if cfg.NETWORK.PRETRAINED else 0
    flag = '-pattern_{}-weight_{}-lr_{}-optim_{}-pretrained_{}_factor_{}'.format(
        cfg.TRAIN.PATTERN, cfg.LOSS.WEIGHT_PRED, cfg.TRAIN.LR, cfg.TRAIN.OPTIMIZER, pretrained, cfg.NETWORK.FACTOR)
    cfg_name += flag

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = final_output_dir / log_file

    fmt = "[%(asctime)s] %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # add file handler to save the log to file
    fh = logging.FileHandler(final_log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    tensorboard_file = 'tensorboard_{}'.format(time_str)
    tensorboard_log_dir = final_output_dir / tensorboard_file
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

# ############## logger related ######################


# ############## dir related ######################
def checkdir(path):
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ############## dir related ######################

# ################ network training #############################
def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


# ################ network training #############################

# ################ save and load #############################

def save_ckpt(model, epoch, optimizer, save_name):
    """Save checkpoint""" 
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    torch.save({
        'epoch': epoch,
        # 'arch': self.model.__class__.__name__,
        'optim_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()}, save_name)

def load_part_params(model, trained_params, ignore):
    params = model.state_dict()
    dict_params = dict(params)

    for n, p in trained_params.items():
        if n in dict_params and ignore not in n:
            dict_params[n].data.copy_(p.data)

    model.load_state_dict(dict_params, strict=False)
    return model

def load_pretrained(model, classifier, output_dir, filename='pretrained/model_p4_baseline_9938_8205_3610.pth.tar'):
    file = filename
    # file = file if os.path.isfile(file) else os.path.join('/apdcephfs/private_haiboqiu/results/occluded-face-recognition-latest/model/fpn/', 'model_p5_w1_9938_9063.pth.tar')
    if os.path.isfile(file):
        logger.info('load pretrained model from {}'.format(file))
        checkpoint = torch.load(file)
        model = load_part_params(model, checkpoint['state_dict'], 'regress')
        # model.load_state_dict(checkpoint['state_dict'], strict=True)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        classifier.load_state_dict(checkpoint['classifier'])

        return model, classifier

    else:
        logger.info('no checkpoint found at {}'.format(file))
        return model, classifier

def load_checkpoint(model, optimizer, classifier, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        classifier.load_state_dict(checkpoint['classifier'])
        logger.info('load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, classifier

    else:
        logger.info('no checkpoint found at {}'.format(file))
        return 0, model, optimizer, classifier

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        state_info = {'state_dict': states['state_dict'],
                      'classifier': states['classifier']}
        torch.save(state_info, os.path.join(output_dir, 'model_best.pth.tar'))

# ################ save and load #############################


# ############### mask and grid related #####################
def occlist_reader(fileList):
    occList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            occPath = line.strip()
            occList.append(occPath)
    return occList

occ_classes = ['book_cover', 'cup', 'eye_mask', 'eyeglasses', 'hand',
           'mouth_mask', 'phone', 'scarf', 'sunglasses']
occ_mapping = {v:(i+1) for v, i in zip(occ_classes, range(len(occ_classes)))}

def occlist_reader_ignore(fileList, idx=0):
    # idx in [0, 1, 2]
    occList = []
    if idx == -1:
        ignores = []
    else:
        ignores = occ_classes[3*idx:3*(idx+1)]

    with open(fileList, 'r') as file:
        for line in file.readlines():
            occPath = line.strip()
            if occPath.split('/')[0] not in ignores:
                occList.append(occPath)
    return occList

def get_occ_type(path):
    cls = os.path.dirname(path)
    if cls in occ_mapping:
        occ_type = occ_mapping[cls]
        return occ_type
    else:
        raise ValueError('unknown occ classes')

def occluded_image_bound(img, occ):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, occ_w), min(H-1, occ_h)
    occ = occ.resize((new_w, new_h))
    
    # in order to the occ within the image, 
    # the random region (for occ left upper point) should be (0, 0, W-new_w, H-new_h)
    x = random.choice(range(0, W-new_w))
    y = random.choice(range(0, H-new_h))
    
    # occlude the img
    box = (x, y, x+new_w, y+new_h)
    img.paste(occ, box)
    
    # cal the corresponding mask
    mask = np.zeros((H, W))
    mask[y:y+new_h, x:x+new_w] = 1.0
    return img, mask

def occluded_image_bound_factor(img, occ, factor=1.0):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, int(factor * occ_w)), min(H-1, int(factor * occ_h))
    occ = occ.resize((new_w, new_h))
    
    # in order to the occ within the image, 
    # the random region (for occ left upper point) should be (0, 0, W-new_w, H-new_h)
    x = random.choice(range(0, W-new_w))
    y = random.choice(range(0, H-new_h))
    
    # occlude the img
    box = (x, y, x+new_w, y+new_h)
    img.paste(occ, box)
    
    # cal the corresponding mask
    mask = np.zeros((H, W))
    mask[y:y+new_h, x:x+new_w] = 1.0
    ratio = (new_h * new_w) / float(H * W)
    return img, mask, ratio

def occluded_image(img, occ):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, occ_w), min(H-1, occ_h)
    occ = occ.resize((new_w, new_h))

    center_x = random.choice(range(0, W))
    center_y = random.choice(range(0, H))
    
    # x = random.choice(range(0, W-new_w))
    # y = random.choice(range(0, H-new_h))
    start_x = center_x - new_w // 2
    start_y = center_y - new_h // 2
    
    end_x = center_x + (new_w + 1) // 2
    end_y = center_y + (new_h + 1) // 2
    # occlude the img
    # box = (x, y, x+new_w, y+new_h)
    box = (start_x, start_y, end_x, end_y)
    img.paste(occ, box)
    
    # cal the corresponding mask
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W-1, end_x)
    end_y = min(H-1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0
    return img, mask

def occluded_image_ratio(img, occ, factor=1.0):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, int(factor * occ_w)), min(H-1, int(factor * occ_h))
    occ = occ.resize((new_w, new_h))

    center_x = random.choice(range(0, W))
    center_y = random.choice(range(0, H))
    
    start_x = center_x - new_w // 2
    start_y = center_y - new_h // 2
    
    end_x = center_x + (new_w + 1) // 2
    end_y = center_y + (new_h + 1) // 2
    # occlude the img
    # box = (x, y, x+new_w, y+new_h)
    box = (start_x, start_y, end_x, end_y)
    img.paste(occ, box)
    
    # cal the corresponding mask
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W-1, end_x)
    end_y = min(H-1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0

    ratio = ((end_y - start_y) * (end_x - start_x)) / float(H * W)

    return img, mask, ratio


def occluded_image_center(img, occ, center):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, occ_w), min(H-1, occ_h)
    occ = occ.resize((new_w, new_h))

    center_x, center_y = center
    
    start_x = center_x - new_w // 2
    start_y = center_y - new_h // 2
    
    end_x = center_x + (new_w + 1) // 2
    end_y = center_y + (new_h + 1) // 2
    # occlude the img
    # box = (x, y, x+new_w, y+new_h)
    box = (start_x, start_y, end_x, end_y)
    img.paste(occ, box)
    
    # cal the corresponding mask
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W-1, end_x)
    end_y = min(H-1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0
    return img, mask

def occluded_image_box(img, occ, box):
    W, H = img.size
    occ_w, occ_h = box[2]-box[0], box[3]-box[1]

    new_w, new_h = min(W-1, occ_w), min(H-1, occ_h)
    occ = occ.resize((new_w, new_h))

    # occlude the img
    # box = (x, y, x+new_w, y+new_h)
    img.paste(occ, box)
    start_x, start_y, end_x, end_y = box
    
    # cal the corresponding mask
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W-1, end_x)
    end_y = min(H-1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0
    ratio = ((end_y - start_y) * (end_x - start_x)) / float(H * W)
    return img, mask, ratio

def get_grids(H, W, N):
    grid_ori = np.zeros((H, W))

    x_axis = np.linspace(0, W, N+1, True, dtype=int)
    y_axis = np.linspace(0, H, N+1, True, dtype=int)

    vertex_set = []
    for y in y_axis:
        for x in x_axis:
            vertex_set.append((y, x))

    grids = [grid_ori]
    for start in vertex_set:
        for end in vertex_set:
            if end[0] > start[0] and end[1] > start[1]:
                grid = grid_ori.copy()
                grid[start[0]:end[0], start[1]:end[1]] = 1.0
                grids.append(grid)
    return grids

def cal_IoU(mask1, mask2):
    inter = np.sum(mask1 * mask2)
    union = np.sum(np.clip(mask1 + mask2, 0, 1)) + 1e-10
    return inter / union

def cal_similarity_label(grids, mask):
    scores = []
    for i, grid in enumerate(grids):
        score = cal_IoU(grid, mask)
        scores.append(score)
    occ_label = np.argmax(scores)
    return occ_label

def soft_binary(mask, thres=0.25):
    N, C, H, W = mask.size()
    mask_flat = mask.reshape(N, C*H*W)

    value, index = mask_flat.sort()
    thres_array = value[:, int(thres*C*H*W)]

    ones = torch.ones_like(mask)
    thres_array = ones.permute(1, 2, 3, 0) * thres_array
    thres_array = thres_array.permute(3, 0, 1, 2)

    assert thres_array.size() == mask.size()
    zeros = torch.zeros_like(mask)
    mask = torch.where(mask > thres_array, mask, zeros)
    return mask

# ############### mask and grid related #####################

if __name__ == '__main__':
    img_size = (112, 96)
    row = 4
    grids = get_grids(*img_size, row)
