import os
import time
import six
import lmdb
import pyarrow as pa
from PIL import Image
from tqdm import tqdm

import pdb

WebFace_LMDB = 'data/datasets/faces_webface_112x112/temp/train.lmdb'
WebFace_Images = 'data/datasets/faces_webface_112x112/images/'
WebFace_List = 'data/datasets/faces_webface_112x112/images.txt'
Occluders = 'data/datasets/occluder/'
Occluders_List = 'data/datasets/occluder/occluder.txt'

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def imglist_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

def generate_file_list(root, write_path):
    file_list = []
    for i, class_name in enumerate(sorted(os.listdir(root))):
        for img_name in sorted(os.listdir(os.path.join(root, class_name))):
            path = os.path.join(class_name, img_name) + ' ' + str(i) + '\n'
            print(path)
            file_list.append(path)
    print(len(file_list))
    with open(write_path, 'w') as f:
        f.writelines(file_list)

def checkdir(path):
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

def buf2img(imgbuf):
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img

def img2buf(img):
    imgbuf = six.BytesIO()
    img.save(imgbuf, format='jpeg')
    imgbuf = imgbuf.getvalue()
    return imgbuf

def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

if __name__ == '__main__':
    # imglist is like '00045/001.jpg'
    # occlist is like 'cup/0093.png'
    # generate_file_list(root='data/datasets/faces_webface_112x112/images/', write_path='data/datasets/faces_webface_112x112/images.txt')
    imgList = imglist_reader(WebFace_List)

    checkdir(WebFace_LMDB)
    lmdb_path = WebFace_LMDB
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, item in tqdm(enumerate(imgList)):
        imgPath, label = item

        img = raw_reader(os.path.join(WebFace_Images, imgPath))
        # pdb.set_trace()

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((img, label, imgPath)))
        if idx % 5000 == 0:
            # time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            # print('{}: {}/{}'.format(time_cur, idx, len(imgList)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
