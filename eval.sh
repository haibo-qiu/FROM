python -u test.py --cfg experiments/CASIA-112x96-LMDB-Mask.yaml \
                    --model 'LResNet50E_IR_FPN'\
                    --batch_size 128 \
                    --gpus '0' \
                    --ratio 3 \
                    --pattern 5 \
                    --debug 1

# ar face dataset testing
#python -u lib/core/ar_dataset.py --protocol 2

# generate npy files for megaface testing
#path='data/datasets/megaface/facescrub_images_factor1.0'
#python -u lib/core/megaface_mp.py --facescrub-root $path\
                                  #--megaface-flag 0 \
                                  #--gpus '1,2,3' \
                                  #-j 1
