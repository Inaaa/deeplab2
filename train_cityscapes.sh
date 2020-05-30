set -e
export PYTHONPATH=/home/chli/cc_code2/deeplab/env/lib/python3.6/site-packages
# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
#CURRENT_DIR=$(pwd)
#WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

# Train 10 iterations.
NUM_ITERATIONS=100000
python train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=8 \
    --dataset="cityscapes" \
    --tf_initial_checkpoint="/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06/deeplabv3_cityscapes_train/model.ckpt" \
    --train_logdir="/mrtstorage/users/students/chli/cityscapes/exp/train_on_train_set/train_fine" \
    --dataset_dir="/mrtstorage/users/students/chli/cityscapes/tfrecord"


# \

