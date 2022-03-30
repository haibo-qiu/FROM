# End2End Occluded Face Recognition by Masking Corrupted Features
This is the Pytorch implementation of our TPAMI 2021 paper [End2End Occluded Face Recognition by Masking Corrupted Features](https://arxiv.org/abs/2108.09468). 
<br>Haibo Qiu, Dihong Gong, Zhifeng Li, Wei Liu and Dacheng Tao<br>

## Requirements
Main packages:
- python=3.6.7
- pytorch=1.8.1
- torchvision=0.9.1
- cudatoolkit=10.2.89
- lmdb=1.2.0
- pyarrow=0.17.0

Or directly create a conda env with
  ```
  conda env create -f environment.yml
  ```

## Data preparation
1. Training data (`data/datasets`) and pretrained models (`pretrained/`) can be found here ([Google drive](https://drive.google.com/drive/folders/12r0QEQFb8MOxh1ZtX679Pnx4g8hknLOg?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1VjuE1nqfytiTYjYWuxP7rA):76n5).

2. Please refer to `data/generate_lmdb.py` for the lmdb file generation of training data.

3. Please refer to `data/generate_occ_lfw.py` for the occluded testing images generation.

## Training
Simply run the following script:
  ```
  bash start.sh
  ```

## Testing
1. To reproduce the results in our paper, please download the [pretrained models](https://drive.google.com/drive/folders/12r0QEQFb8MOxh1ZtX679Pnx4g8hknLOg?usp=sharing) and put them in `pretrained/`, then run:
    ```
    bash eval.sh
    ```
2. For megaface testing, the related commonds are included in `eval.sh`. Current `lib/core/megaface_mp.py` generates npy file for each sample, which can be evaluated with [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/test_protocol/megaface). Or you can switch the generated function in `lib/core/megaface_mp.py` to produce bin file and use [official devkit](https://megaface.cs.washington.edu/participate/challenge.html) for evaluation. 

3. The AR Face dataset evaluation scripts are also included in `eval.sh`.

## Acknowledgement
The code is partially developed from [PDSN](https://github.com/linserSnow/PDSN). The [occluders images](https://drive.google.com/drive/folders/12r0QEQFb8MOxh1ZtX679Pnx4g8hknLOg?usp=sharing) are also from [PDSN](https://github.com/linserSnow/PDSN).

## Citation
If you use our code or models in your research, please cite with:
```
@article{qiu2021end2end,
  title={End2End occluded face recognition by masking corrupted features},
  author={Qiu, Haibo and Gong, Dihong and Li, Zhifeng and Liu, Wei and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
