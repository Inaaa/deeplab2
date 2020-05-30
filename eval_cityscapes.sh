set -e

# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
#CURRENT_DIR=$(pwd)
#WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

# Train 10 iterations.
#NUM_ITERATIONS=100000

python eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --dataset_dir = "/mrtstorage/users/students/chli/cityscapes/tfrecord" \
    --checkpoint_dir="/mrtstorage/users/students/chli/cityscapes/exp/train_on_train_set/train_fine" \
    --eval_logdir="/mrtstorage/users/students/chli/cityscapes/exp/train_on_train_set/eval_fine"
    #--checkpoint_dir="/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06/deeplabv3_cityscapes_train/"\
    #--eval_logdir = "/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06/eval"
    #--dataset_dir="/mrtstorage/users/students/chli/cityscapes/tfrecord"\

