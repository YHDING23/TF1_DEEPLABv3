## A Tensorflow 1.15 Implementation of Training Deeplab v3 via Cityscapes Dataset

This script is for training the a Deeplabv3-Semantic-Segmentation model using Cityscapes dataset. 

Reference: [Running DeepLab on Cityscapes semantic segmentation dataset.](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md)

DL framework is:
- Tensorflow-gpu==1.15.0
- Python == 3.6

### 1 - Setup
- Clone the repo, setup the virtual env
```angular2html

git clone https://github.com/tensorflow/models.git
cd models/research/deeplab

mkdir venv
virtualenv -p python3.6 venv
source venv/bin/activate
install -r requirements.txt
```

### 2 - Prepare the dataset. 
We have a copy of TFRecord of cityscapes located in our NFS server `nfs_3/tf_records/cityscapes_tfrecord`. If you are using the same capy, skip this section. 

If you are using the original images of cityscapes, check the file [convert_cityscapes.sh](https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/convert_cityscapes.sh) The folder structure is assumed to be:
```angular2html
+ datasets
#    - build_cityscapes_data.py
#    - convert_cityscapes.sh
#    + cityscapes
#      + cityscapesscripts (downloaded scripts)
#      + gtFine
#      + leftImg8bit

```
Regarding this scirpt, only three major folders are used: `cityscapesscripts/`, `gtFine/` and `leftImg8bit`.  Otherwise, you can download the data from the official website. 

- Covert the dataset to TFRecord, and it takes 15 mins
```angular2html
sh convert_cityscapes.sh
```
### 3 - Requirements

Reference at https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md

```angular2html

# From TF1_Deeplabv3/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### 4 - Pickup model and Train

Reference at https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

Example: I pickup the pretrained checkpoint name: `xception65_coco_voc_trainaug`.
```angular2html
# from tensorflow/models/research/deeplab
mkdir log
mkdir pretrain
cd pretrain
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
tar -xf deeplabv3_pascal_train_aug_2018_01_04.tar.gz
```

If you are using the TFRecord in `/nfs_3`, the training flags are as followings:
```angular2html
# from directory "TF1_Deeplabv3"

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=1 \
    --dataset="cityscapes" \
    --checkpoint_dir=pretrain/deeplabv3_pascal_train_aug \
    --train_logdir=log \
    --dataset_dir=/nfs_3/tf_records/cityscapes_tfrecord/

```
