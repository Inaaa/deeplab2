set -e

export PYTHONPATH=/home/chli/cc_code2/deeplab/env/lib/python3.6/site-packages


python export_model.py \
    --checkpoint_path=/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train2/model.ckpt-0 \
    --export_path=/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train2/frozen_inference_graph.pb \
    --model_variant="xception_65"\
    --atrous_rates=6   \
    --atrous_rates=12  \
    --atrous_rates=18  \
    --output_stride=16 \
    --decoder_output_stride=4  \
    --num_classes=19  \
    --inference_scales=1.0 \
    --input_type image_tensor



