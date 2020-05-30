set -e

# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
#WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

python vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="/mrtstorage/users/students/chli/cityscapes_slope/exp/train_on_train_set/train3" \
    --vis_logdir="/mrtstorage/users/students/chli/cityscapes_slope/exp/train_on_train_set/vis3" \
    --dataset_dir="/mrtstorage/users/chli/cityscapes_slope/tfrecord"
 

