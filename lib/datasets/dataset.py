import os
import six
import lmdb
import torch
import pickle
import random
import numpy as np
import pyarrow as pa
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageDraw
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

import lib.core.utils as utils
from lib.core.config import config
import pdb

Occluders = 'data/datasets/occluder/'
Occluders_List = 'data/datasets/occluder/occluder.txt'
ImageFile.LOAD_TRUNCATED_IAMGES = True

class WebFace_Folder(Dataset):
    def __init__(self, root, mode, img_size=(112, 96), pattern=3, ratio=3, albu_transform=None, transform=None):
        super(WebFace_Folder, self).__init__()
        self.root = root
        self.db, self.class_to_label = self.load_db()
        self.classes = list(self.class_to_label.keys())

        self.occList = utils.occlist_reader(Occluders_List)
        self.occRoot = Occluders
        self.grids = utils.get_grids(*img_size, pattern)
        self.img_size = img_size

        self.mode = mode
        self.ratio = ratio
        self.albu_transform = albu_transform
        self.transform = transform

    def load_db(self):
        db = []
        class_to_label = {}
        for i, class_name in enumerate(sorted(os.listdir(self.root))):
            if class_name not in class_to_label:
                class_to_label[class_name] = i

            for img_name in (os.listdir(os.path.join(self.root, class_name))):
                datum = [os.path.join(class_name, img_name), i]
                db.append(datum)

        return db, class_to_label

    def PIL_loader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def occlude_img(self, img):
        # occlude img
        occPath = random.choice(self.occList)
        occ = self.PIL_loader(os.path.join(self.occRoot, occPath))
        img_occ, mask = utils.occluded_image(img.copy(), occ)

        # cal mask label
        mask_label = utils.cal_similarity_label(self.grids, mask)
        return img_occ, mask_label, mask

    def get_data(self, img):
        if self.mode == 'Clean':
            return img, 0

        elif self.mode == 'Occ':
            img_occ, mask_label, mask = self.occlude_img(img)
            return img_occ, mask_label

        elif self.mode == 'Mask':
            if random.choice(range(self.ratio)) == 0:
                return img, 0
            else:
                img_occ, mask_label, mask = self.occlude_img(img)
                return img_occ, mask_label
        else:
            raise ValueError('unknown mode!')

    def __getitem__(self, index):
        datum = self.db[index]
        img_path, label = datum

        img = self.PIL_loader(os.path.join(self.root, img_path))
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = random.choice([img, img_flip])

        img, mask_label = self.get_data(img)

        if self.albu_transform is not None:
            img_np = np.array(img)
            augmented = self.albu_transform(image=img_np)
            img = Image.fromarray(augmented['image'])

        if self.transform is not None and self.mode != 'Mask':
            size = random.choice([96, 112, 128])
            new_transform = transforms.Compose([transforms.Resize(size),
                                                self.transform])
            img = new_transform(img)
        else:
            img = self.transform(img)

        return img, label, mask_label, img_path

    def __len__(self):
        return len(self.db)

class WebFace_LMDB(Dataset):
    def __init__(self, db_path, mode, img_size=(112, 96), pattern=3, transform=None, ratio=3):
        super(WebFace_LMDB, self).__init__()
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.occList = utils.occlist_reader(Occluders_List)
        # self.occList = utils.occlist_reader_ignore(Occluders_List, idx=ignore_idx)
        self.occRoot = Occluders
        self.grids = utils.get_grids(*img_size, pattern)
        self.img_size = img_size

        self.mode = mode
        self.ratio = ratio
        self.transform = transform

    def PIL_reader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def buf2img(self, imgbuf):
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def img2buf(self, img):
        imgbuf = six.BytesIO()
        img.save(imgbuf, format='jpeg')
        imgbuf = imgbuf.getvalue()
        return imgbuf

    def occlude_img(self, img):
        # occlude img
        occPath = random.choice(self.occList)
        occ = self.PIL_reader(os.path.join(self.occRoot, occPath))
        factor = random.choice(np.linspace(1, 5, 9, endpoint=True))
        img_occ, mask, _ = utils.occluded_image_ratio(img.copy(), occ, factor)

        # cal mask label
        mask_label = utils.cal_similarity_label(self.grids, mask)
        return img_occ, mask_label, mask

    def get_data(self, img):
        if self.mode == 'Clean':
            return img, 0

        # elif self.mode == 'Occ':
            # img_occ, mask_label, mask = self.occlude_img(img)
            # return img_occ, mask_label

        elif self.mode == 'Mask' or self.mode == 'Occ':
            if random.choice(range(self.ratio)) == 0:
                return img, 0
            else:
                img_occ, mask_label, mask = self.occlude_img(img)
                return img_occ, mask_label
        else:
            raise ValueError('unknown mode!')

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        imgbuf, label, img_path = unpacked

        # load image
        img = self.buf2img(imgbuf)
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = random.choice([img, img_flip])
        # pdb.set_trace()

        img, mask_label = self.get_data(img)

        if self.transform is not None and self.mode == 'Clean':
            size = random.choice([96, 112, 128])
            new_transform = transforms.Compose([transforms.Resize(size),
                                                self.transform])
            img = new_transform(img)
        else:
            img = self.transform(img)

        return img, label, mask_label, img_path

    def vis_samples(self, num, writer, shuffle=True):
        assert num < self.length
        if shuffle:
            indexList = random.choices(range(self.length), k=num)
        else:
            indexList = range(num)

        Tensor2PIL = transforms.ToPILImage()
        PIL2Tensor = transforms.ToTensor()

        grid_imgs = torch.zeros(num, 3, *self.img_size)
        for i, idx in enumerate(indexList):
            img, label, mask_label = self.__getitem__(idx)
            img = Tensor2PIL(img)
            img_draw = ImageDraw.Draw(img)
            # (w_start, h_start, w_end, h_end)
            img_draw.rectangle((0, 0, 50, 10), fill=(150, 150, 150))
            img_draw.text((0, 0), '{}/{}'.format(label, mask_label), fill=(255, 255, 255))
            grid_imgs[i] = PIL2Tensor(img)

        writer.add_images('vis_imgs_labels', grid_imgs, 0)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class LFW_Image(Dataset):
    def __init__(self, config, transform=None):
        super(LFW_Image, self).__init__()
        self.lfw_path = config.DATASET.LFW_PATH
        self.lfw_occ_path = config.DATASET.LFW_OCC_PATH
        self.num_class = config.DATASET.LFW_CLASS
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.LFW_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            # if os.path.exists(self.lfw_occ_path + name1) and os.path.exists(self.lfw_occ_path + name2):
            if os.path.exists(self.lfw_occ_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        # p = self.pairs[index].replace('\n', '').split('\t')
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        with open(self.lfw_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')

        with open(self.lfw_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')

        with open(self.lfw_occ_path + name2, 'rb') as f:
            img2_occ =  Image.open(f).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class O_LFW_Image(Dataset):
    def __init__(self, config, transform=None):
        super(O_LFW_Image, self).__init__()
        self.olfw_path = config.DATASET.OLFW_PATH
        self.num_class = config.DATASET.LFW_CLASS
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.OLFW_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split('\t')
            # pdb.set_trace()
            name1, name2, sameflag = p

            if os.path.exists(self.olfw_path + name1) and os.path.exists(self.olfw_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        name1, name2, sameflag = self.pairs[index]
        sameflag = int(sameflag)

        with open(self.olfw_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')

        with open(self.olfw_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class MFR2_Image(Dataset):
    def __init__(self, config, transform=None):
        super(MFR2_Image, self).__init__()
        self.mfr2_path = config.DATASET.MFR2_PATH
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.MFR2_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.strip().split(' ')
            # pdb.set_trace()

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            if os.path.exists(self.mfr2_path + name1) and os.path.exists(self.mfr2_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        with open(self.mfr2_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')

        with open(self.mfr2_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2 
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class RMFD_Image(Dataset):
    def __init__(self, config, transform=None):
        super(RMFD_Image, self).__init__()
        self.rmfd_path = config.DATASET.RMFD_PATH
        self.mode = config.TEST.MODE 
        self.pairs = self.get_pairs_lines(config.DATASET.RMFD_PAIRS)

        self.transform = transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.strip().split(' ')
            # pdb.set_trace()
            assert len(p) == 3
            name1, name2, sameflag = p

            if os.path.exists(self.rmfd_path + name1) and os.path.exists(self.rmfd_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        name1, name2, sameflag = self.pairs[index]
        sameflag = int(sameflag)

        with open(self.rmfd_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')

        with open(self.rmfd_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = img2
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class Megaface_Image(Dataset):
    def __init__(self, input_dir, transform=None, occ_box=None, jitter=None):
        super(Megaface_Image, self).__init__()
        self.db = self.get_dataset_pkl(input_dir)

        self.transform = transform
        self.occ_box = occ_box
        self.jitter = jitter
        self.root = input_dir

    def get_dataset_pkl(self, input_dir):
        print('loading from {}...'.format(input_dir))
        pkl_path = input_dir + '.pkl'
        print(pkl_path)
        assert os.path.exists(pkl_path), 'invalid dataset!'
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        return db

    def get_dataset_facescrub(self, input_dir):
        print('loading from {}, it will cost about 2s'.format(input_dir))
        ret = []
        label = 0
        person_names = sorted(os.listdir(input_dir))
        for person_name in person_names:
            subdir = os.path.join(input_dir, person_name)
            if not os.path.isdir(subdir):
                print('{} does not exist'.format(subdir))
                continue
            for img in os.listdir(subdir):
                fimage = edict()
                fimage.id = os.path.join(person_name, img)
                fimage.classname = str(label)
                fimage.image_path = os.path.join(subdir, img)
                ret.append(fimage)
                ret.append(fimage)
            label += 1
        return ret

    def get_dataset_megaface(self, input_dir):
        print('loading from {}, it will cost about 3 mins'.format(input_dir))
        ret = []
        label = 0
        count = 0
        for prefixdir in os.listdir(input_dir):
            len1 = len(os.listdir(input_dir))
            count += 1
            _prefixdir = os.path.join(input_dir, prefixdir)
            for subdir in os.listdir(_prefixdir):
                _subdir = os.path.join(_prefixdir, subdir)
                if not os.path.isdir(_subdir):
                    print('{} does not exist'.format(_subdir))
                    continue
                # if label % 1000 == 0:
                    # print('{:03}/{}, {}, {}'.format(count, len1, label, _subdir))
                for img in os.listdir(_subdir):
                    fimage = edict()
                    fimage.id = os.path.join(prefixdir, subdir, img)
                    fimage.classname = str(label)
                    fimage.image_path = os.path.join(_subdir, img)
                    ret.append(fimage)
                    ret.append(fimage)
                label += 1
        return ret

    def occ_image(self, img):
        occ_box = self.occ_box
        occList=self.occlist_reader(Occluders_List)
        occPath = random.choice(occList)
        with open(os.path.join(Occluders, occPath), 'rb') as f:
            occ = Image.open(f).convert('RGB')
        if self.jitter:
            # jitter = random.choice(range(-3, 4))
            # occ_box = tuple([v+jitter for v in occ_box])
            jitter = random.choice(range(-5, 6))
            occ_box = [v+jitter for v in occ_box]
            H, W = 112, 96
            occ_box[0] = occ_box[0] if occ_box[0] >= 0 else 0
            occ_box[1] = occ_box[1] if occ_box[1] >= 0 else 0
            occ_box[2] = occ_box[2] if occ_box[2] < W else W-1
            occ_box[3] = occ_box[3] if occ_box[3] < H else H-1
            occ_box = tuple(occ_box)
        new_w, new_h = (occ_box[2]-occ_box[0], occ_box[3]-occ_box[1])
        occ = occ.resize((new_w, new_h))
        img.paste(occ, occ_box)
        return img

    def occlist_reader(self, fileList):
        occList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                occPath = line.strip()
                occList.append(occPath)
        return occList

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        fimage = self.db[index]
        img_path = os.path.join(os.path.dirname(self.root), fimage.image_path)

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if index % 2 == 1:
            img = transforms.functional.hflip(img) 

        if self.occ_box:
            img_occ =self.occ_image(img.copy())
        else:
            img_occ = img.copy()

        if self.transform:
            img = self.transform(img)
            img_occ = self.transform(img_occ)

        return img, img_occ, fimage.id, fimage.classname

class Megaface_Image_MP(Dataset):
    def __init__(self, input_dir, rank, procs, transform=None, occ_box=None, jitter=None):
        super(Megaface_Image_MP, self).__init__()
        self.db = self.get_dataset_pkl(input_dir, rank, procs)

        self.transform = transform
        self.occ_box = occ_box
        self.jitter = jitter
        self.root = input_dir

    def get_dataset_pkl(self, input_dir, rank, procs):
        utils.mp_print('loading from {}...'.format(input_dir))
        pkl_path = input_dir + '_single.pkl'
        utils.mp_print(pkl_path)
        assert os.path.exists(pkl_path), 'invalid dataset!'
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        return db[rank::procs]

    def occ_image(self, img):
        occ_box = self.occ_box
        occList=self.occlist_reader(Occluders_List)
        occPath = random.choice(occList)
        with open(os.path.join(Occluders, occPath), 'rb') as f:
            occ = Image.open(f).convert('RGB')
        if self.jitter:
            # jitter = random.choice(range(-3, 4))
            # occ_box = tuple([v+jitter for v in occ_box])
            jitter = random.choice(range(-10, 11))
            occ_box = [v+jitter for v in occ_box]
            H, W = 112, 96
            occ_box[0] = occ_box[0] if occ_box[0] >= 0 else 0
            occ_box[1] = occ_box[1] if occ_box[1] >= 0 else 0
            occ_box[2] = occ_box[2] if occ_box[2] < W else W-1
            occ_box[3] = occ_box[3] if occ_box[3] < H else H-1
            occ_box = tuple(occ_box)
        new_w, new_h = (occ_box[2]-occ_box[0], occ_box[3]-occ_box[1])
        occ = occ.resize((new_w, new_h))
        img.paste(occ, occ_box)
        return img

    def occlist_reader(self, fileList):
        occList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                occPath = line.strip()
                occList.append(occPath)
        return occList

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        fimage = self.db[index]
        img_path = os.path.join(os.path.dirname(self.root), fimage.image_path)

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.occ_box:
            img_occ =self.occ_image(img.copy())
            img_occ_flip =self.occ_image(img_flip.copy())
        else:
            img_occ = img.copy()
            img_occ_flip = img_flip.copy()

        if self.transform:
            img = self.transform(img)
            img_flip = self.transform(img_flip)
            img_occ = self.transform(img_occ)
            img_occ_flip = self.transform(img_occ_flip)

        # return img, img_flip, img_occ, img_occ_flip, fimage.id, fimage.classname
        return img, img_flip, fimage.id

class ARdataset(Dataset):
    def __init__(self, input_dir, transform=None):
        super(ARdataset, self).__init__()
        self.db = self.get_ardata(input_dir)
        self.transform = transform

    def generate_new_gallery(self, input_dir):
        new_dir = input_dir.replace('gallery', 'gallery_single')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        img_names = os.listdir(input_dir)
        img_names.sort()
        new_names = img_names[::6]
        print(len(new_names), new_names)
        for name in new_names:
            print(name)
            with open(os.path.join(input_dir, name), 'rb') as f:
                img = Image.open(f).convert('RGB')
            img.save(os.path.join(new_dir, name))


    def get_ardata(self, input_dir):
        print('loading from {}'.format(input_dir))
        ret = []
        img_names = os.listdir(input_dir)
        img_names.sort()
        for img_name in img_names:
            img_path = os.path.join(input_dir, img_name)
            gender = img_name.split('-')[0]
            num = img_name.split('-')[1]
            label = '{}_{}'.format(gender, num)

            fimage = edict()
            fimage.label = label
            fimage.image_path = img_path
            ret.append(fimage)
            ret.append(fimage)
        return ret

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        fimage = self.db[index]

        with open(fimage.image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if index % 2 == 1:
            img = transforms.functional.hflip(img) 

        if self.transform:
            img = self.transform(img)

        return img, fimage.label

def get_occ_boxs():
    # unchange
    left_face = (12, 44, 52, 104)
    right_face = (52, 44, 92, 104)
    up_face = (10, 15, 85, 65)
    down_face = (10, 65, 85, 105)
    twoeyes = (16, 44, 86, 64)
    
    # hard
    left_eye_hard = (10, 35, 50, 65)
    right_eye_hard = (50, 35, 90, 65)
    nose_hard = (33, 45, 63, 85)
    mouth_hard = (25, 80, 75, 110)
    nose_mouth_hard = (25, 60, 75, 110)
    boxes_hard = [left_face, right_face, up_face, down_face, twoeyes, left_eye_hard, right_eye_hard, nose_hard, mouth_hard, nose_mouth_hard]
    names_hard = ['left_face', 'right_face', 'up_face', 'down_face', 'twoeyes', 'left_eye_hard', 'right_eye_hard', 'nose_hard', 'mouth_hard', 'nose_mouth_hard']

    return boxes_hard, names_hard

if __name__ == '__main__':
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    occ_boxes, occ_names = get_occ_boxs()

    writer = SummaryWriter('temp')
    input_dir = 'data/datasets/megaface/facescrub_images'
    for occ_box, occ_name in zip(occ_boxes, occ_names):
        print(occ_name, occ_box)
        test_loader = torch.utils.data.DataLoader(
            Megaface_Image(input_dir, test_transform, occ_box),
            batch_size=512, 
            shuffle=True,
            num_workers=8, 
            pin_memory=True)

        grid_image = np.zeros((512, 112, 96, 3))
        for batch_idx, (img, img_occ, identity, label) in enumerate(test_loader):
            print(batch_idx)
            img =img_occ
            img = img.numpy()
            img = img.transpose(0, 2, 3, 1)
            grid_image = (img - np.min(img)) / float(np.ptp(img))
            writer.add_images('vis_occ_imgs_{}'.format(occ_name), grid_image, 0, dataformats='NHWC')
            break
    writer.close()

