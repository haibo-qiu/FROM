import os
import sys
import time
import json
import numpy as np
import argparse
import struct
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append('./')
import lib.core.utils as utils
from lib.core.config import config
from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN
from lib.models.fpn import LResNet50E_IR_Occ_2D
from lib.models.fpn import LResNet50E_IR_Occ_FC
from lib.datasets.dataset import Megaface_Image
from lib.datasets.dataset import Megaface_Image_MP

def mp_print(msg):
    if dist.get_rank() == 0:
        print(msg)

DATA_ROOT = 'data/datasets/'
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo', type=str, help='', default='occface')
    parser.add_argument('--facescrub-root', type=str, help='', default=DATA_ROOT + 'megaface/facescrub_images_occ')
    parser.add_argument('--megaface-root', type=str, help='',default=DATA_ROOT + 'megaface/megaface_images')
    parser.add_argument('--megaface-flag', type=int, help='',default=0)
    parser.add_argument('--output', type=str, help='', default='megaface/')
    parser.add_argument('--model', type=str, help='', default='')
    parser.add_argument("-j", "--procs", help = "number of procs per gpu", default = 8, type = int)
    parser.add_argument("--dist-url", help = "init_method for distributed training", default = 'tcp://127.0.0.1:8081', type = str)
    parser.add_argument("--dist-backend", help = "distributed backend", default = 'nccl', type = str)
    parser.add_argument('--gpus', help='gpus', type=str)
    return parser.parse_args()

def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature),1,4,5))
        f.write(struct.pack("%df"%len(feature), *feature))

def get_models_params():
    models_root = 'pretrained/'
    models_names = ['model_p5_w1_9938_9063.pth.tar']

    models_params = []
    for name in models_names:
        model_path = os.path.join(models_root, name)
        assert os.path.exists(model_path), 'invalid model name!'

        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        # model.module.load_state_dict(state_dict)

        model_name = name.split('.')[0]
        models_params.append((model_name, state_dict))
    return models_params

def extractDeepFeature(img, model, mask=None):
    img = img.to('cuda')
    fc_mask, mask, vec, fc = model(img, mask)
    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    
    return fc, fc_mask, mask

def get_fc(img, model, model_name):
    fc, fc_mask, mask = extractDeepFeature(img, model)

    if 'baseline' not in model_name and '_occ_' not in model_name:
        fc = fc_mask
    return fc

def get_feature(img, img_flip, model, model_name):
    assert img.size() == img_flip.size()

    imgs = torch.cat((img, img_flip), dim=0)
    fc = get_fc(imgs, model, model_name)

    fc1 = fc[:img.size(0), :]
    fc2 = fc[img.size(0):, :]

    fc = fc1 + fc2

    fc = F.normalize(fc)
    return fc

def write_npy(identity, features, dataset_out, model_name, previous_iter, length, rank):
    for i, _iden in enumerate(identity):
        fc = features[i].flatten()
        fc = fc.detach().numpy()

        # if dataset_out.endswith('facescrub') or dataset_out.endswith('facescrub_occ'):
        if 'facescrub' in dataset_out.split('/')[-1]:
            pre, post = _iden.split('/')
            out_dir = os.path.join(dataset_out, pre)
        elif dataset_out.endswith('megaface'):
            pre, mid, post = _iden.split('/')
            out_dir = os.path.join(dataset_out, pre, mid)
        else:
            raise ValueError('Unknown dataset type!')
        assert os.path.exists(out_dir)

        # out_path = os.path.join(out_dir, post+'_{}.bin'.format(model_name))
        out_path = os.path.join(out_dir, os.path.splitext(post)[0]+'.npy')
        cur_iter = previous_iter + i
        if cur_iter % 1000 == 0:
            time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            mp_print('process[{}], {}, writing {:06}/{}: {}'.format(rank, time_cur, cur_iter, length, out_path))

        np.save(out_path, fc)

def write_feature(identity, features, dataset_out, model_name, previous_iter, length, rank):
    for i, _iden in enumerate(identity):
        fc = features[i].flatten()
        fc = fc.detach().numpy()

        # if dataset_out.endswith('facescrub') or dataset_out.endswith('facescrub_occ'):
        if 'facescrub' in dataset_out.split('/')[-1]:
            pre, post = _iden.split('/')
            out_dir = os.path.join(dataset_out, pre)
        elif dataset_out.endswith('megaface'):
            pre, mid, post = _iden.split('/')
            out_dir = os.path.join(dataset_out, pre, mid)
        else:
            raise ValueError('Unknown dataset type!')
        assert os.path.exists(out_dir)

        out_path = os.path.join(out_dir, post+'_{}.bin'.format(model_name))
        cur_iter = previous_iter + i
        if cur_iter % 1000 == 0:
            time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            mp_print('process[{}], {}, writing {:06}/{}: {}'.format(rank, time_cur, cur_iter, length, out_path))

        write_bin(out_path, fc)

def copydirs(source, dest):
    print('copy dirs from {} to {}'.format(source, dest))
    for cur, dirs, files in tqdm(os.walk(source)):
        cur = cur.split('/')
        new = '/'.join([dest] + cur[len(source.split('/')):])
        if not os.path.exists(new):
            os.makedirs(new)

def handledirs(args):
    postfix_facescrub = args.facescrub_root.split('/')[-1]
    facescrub_out = os.path.join(args.output, postfix_facescrub)
    megaface_out = os.path.join(args.output, 'megaface')
    copydirs(args.facescrub_root, facescrub_out)
    if args.megaface_flag:
        copydirs(args.megaface_root, megaface_out)

def main():
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = [int(i) for i in args.gpus.split(',')]
    args.gpus = gpus
    print(args)
    handledirs(args)
    cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy('file_system')
    nprocs = args.procs * len(args.gpus)

    start = time.time()
    torch.multiprocessing.spawn(test, args=(args, nprocs), nprocs=nprocs, join=True, daemon=False)
    end = time.time()
    print('cost {} hours'.format((end - start) / 3600.))

def test(rank, args, world_size):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank % len(args.gpus))

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    facescrub_loader = torch.utils.data.DataLoader(
        Megaface_Image_MP(args.facescrub_root, rank, world_size, transform),
        batch_size=config.TEST.BATCH_SIZE, 
        shuffle=config.TEST.SHUFFLE,
        num_workers=4, 
        pin_memory=True)
    facescrub_length = len(facescrub_loader.dataset)

    megaface_loader = torch.utils.data.DataLoader(
        Megaface_Image_MP(args.megaface_root, rank, world_size, transform),
        batch_size=config.TEST.BATCH_SIZE, 
        shuffle=config.TEST.SHUFFLE,
        num_workers=4, 
        pin_memory=True)
    megaface_length = len(megaface_loader.dataset)

    postfix_facescrub = args.facescrub_root.split('/')[-1]
    mp_print("process[{}] start: facescrub length:{} megaface length:{}".format(
        rank, facescrub_length, megaface_length))

    for models_params in get_models_params():
        model_name, state_dict = models_params
        # args.output = os.path.join(args.output, model_name)
        # handledirs(args)
        facescrub_out = os.path.join(args.output, postfix_facescrub)
        megaface_out = os.path.join(args.output, 'megaface')
        mp_print('process[{}]: {}'.format(rank, model_name))
        pattern = int(model_name[model_name.find('p')+1])
        num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
        model = LResNet50E_IR_FPN(num_mask=num_mask)
        model.load_state_dict(state_dict, strict=True)
        model = model.cuda()
        model.eval()

        if args.megaface_flag:
            for batch_idx, (img, img_flip, identity) in enumerate(megaface_loader):
                features = get_feature(img, img_flip, model, model_name)
                write_npy(identity, features, megaface_out, model_name, 
                              batch_idx*config.TEST.BATCH_SIZE, megaface_length, rank)
            continue

        for batch_idx, (img, img_flip, identity) in enumerate(facescrub_loader):
            features = get_feature(img, img_flip, model, model_name)
            write_npy(identity, features, facescrub_out, model_name, 
                          batch_idx*config.TEST.BATCH_SIZE, facescrub_length, rank)

if __name__ == '__main__':
    main()

