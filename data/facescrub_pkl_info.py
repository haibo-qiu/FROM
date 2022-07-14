import os
import pickle
import time
from easydict import EasyDict as edict

def get_dataset_facescrub(input_dir):
    print('loading from {}, it will cost about 2s'.format(input_dir))
    ret = []
    label = 0
    person_names = sorted(os.listdir(input_dir))
    for person_name in person_names:
        subdir = os.path.join(input_dir, person_name)
        if not os.path.isdir(subdir):
            print('{} does not exist'.format(subdir))
            continue
        for img in sorted(os.listdir(subdir)):
            fimage = edict()
            fimage.id = os.path.join(person_name, img)
            fimage.classname = str(label)
            fimage.image_path = os.path.join(subdir, img)
            ret.append(fimage)
            # ret.append(fimage)
        label += 1
    return ret

def get_dataset_megaface(input_dir):
    print('loading from {}, it will cost about 3 mins'.format(input_dir))
    ret = []
    label = 0
    count = 0
    start = time.time()
    for prefixdir in sorted(os.listdir(input_dir)):
        len1 = len(os.listdir(input_dir))
        count += 1
        _prefixdir = os.path.join(input_dir, prefixdir)
        for subdir in sorted(os.listdir(_prefixdir)):
            _subdir = os.path.join(_prefixdir, subdir)
            if not os.path.isdir(_subdir):
                print('{} does not exist'.format(_subdir))
                continue
            # if label % 1000 == 0:
                # print('{:03}/{}, {}, {}'.format(count, len1, label, _subdir))
            for img in sorted(os.listdir(_subdir)):
                fimage = edict()
                fimage.id = os.path.join(prefixdir, subdir, img)
                fimage.classname = str(label)
                fimage.image_path = os.path.join(_subdir, img)
                ret.append(fimage)
                # ret.append(fimage)
            label += 1
    end = time.time()
    print('cost {} mins'.format((end - start) / 60.0))
    return ret

def facescrub_dataset(input_lst, root):
    ret = []
    for line in open(input_lst, 'r'):
        image_path = line.strip()
        person_name, img = image_path.split('/')

        fimage = edict()
        fimage.id = os.path.join(person_name, img)
        fimage.classname = person_name
        fimage.image_path = os.path.join(root, image_path)
        print(fimage)
        ret.append(fimage)
    return ret

def megaface_dataset(input_lst, root):
    ret = []
    count = 0
    for line in open(input_lst, 'r'):
        image_path = line.strip()
        person_name, cls, img = image_path.split('/')

        fimage = edict()
        fimage.id = image_path
        fimage.classname = person_name
        fimage.image_path = os.path.join(root, image_path)
        ret.append(fimage)
        if count % 1000 == 0:
            print(count, fimage)
        count += 1
    return ret

def main():
    facescrub = 'facescrub_images'
    facescrub_occ = 'facescrub_images_occ'
    megaface = 'megaface_images'
    datasets = ['facescrub_mask']
    for dataset in datasets:
        # ret = facescrub_dataset('facescrub_lst', dataset)
        ret = get_dataset_facescrub(dataset)
        with open(dataset+'_single.pkl', 'wb') as f:
            pickle.dump(ret, f)

    # ret = get_dataset_megaface(megaface)
    # ret = megaface_dataset('megaface_lst', megaface)
    # with open(megaface+'_simple.pkl', 'wb') as f:
        # pickle.dump(ret, f)

if __name__ == '__main__':
    main()
