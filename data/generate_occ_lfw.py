import os
import sys
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import pdb

sys.path.insert(0, './')
import lib.core.utils as utils

random.seed(100)

data_path = 'data/datasets/lfw-occ/lfw-112X96'
# data_save = 'data/datasets/lfw/lfw-112X96-new-occ'
# data_path = 'data/datasets/megaface/facescrub_images'
# data_save = 'data/datasets/megaface/facescrub_images_occ'
Occluders = 'data/datasets/occluder/'
Occluders_List = 'data/datasets/occluder/occluder.txt'

def get_occ_boxs_v2():
    # unchange
    left_eye = (10, 30, 50, 70)
    right_eye = (50, 30, 90, 70)
    twoeyes = (10, 30, 90, 70)
    nose = (34, 48, 62, 86)
    mouth = (25, 82, 75, 107)
    down_face = (8, 60, 92, 110)
    up_face = (8, 8, 92, 72)
    left_face = (8, 8, 50, 100)
    right_face = (50, 8, 92, 100)

    boxes = [left_eye, right_eye, twoeyes, nose, mouth, down_face, up_face, left_face, right_face]
    names = ['left_eye', 'right_eye', 'twoeyes', 'nose', 'mouth', 'down_face', 'up_face', 'left_face', 'right_face']
    
    return boxes, names

def PIL_Reader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def main():
    occList = utils.occlist_reader(Occluders_List)

    boxes, names = get_occ_boxs_v2()
    names = boxes = [2.0]
    # factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    for box, name in zip(boxes, names):
        factor = box
        data_save = data_path + '-occ{}-temp'.format(factor)
        # data_save = data_path + '_random_factor'
        print(data_save)
        ratios = []

        for i, subdir in tqdm(enumerate(os.listdir(data_path))):
            for img_name in os.listdir(os.path.join(data_path, subdir)):
                img_path = os.path.join(data_path, subdir, img_name)
                # print('{} / {}, process {}...'.format(i, len(os.listdir(data_path)), img_path))
                img = PIL_Reader(img_path)

                occ_path = random.choice(occList)
                occ = PIL_Reader(os.path.join(Occluders, occ_path))
                # print('IMG-[{}] is occluded by OCC-[{}]'.format(img_path, occ_path))

                img_occ, mask, ratio = utils.occluded_image_ratio(img, occ, factor)
                # img_occ, mask, ratio = utils.occluded_image_box(img, occ, box)
                ratios.append(ratio)
                img_save = os.path.join(data_save, subdir, img_name)
                utils.checkdir(img_save)
                img_occ.save(img_save)
        print(np.mean(ratios))

if __name__ == '__main__':
    main()
