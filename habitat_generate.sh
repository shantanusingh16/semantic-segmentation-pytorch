#!/bin/bash

# model names
MODEL_NAME=ade20k-mobilenetv2dilated-c1_deepsup
MODEL_PATH=ckpt/$MODEL_NAME
RESULT_PATH=./

ENCODER=$MODEL_NAME/encoder_epoch_20.pth
DECODER=$MODEL_NAME/decoder_epoch_20.pth

# Download model weights and image
# if [ ! -e $MODEL_PATH ]; then
#   mkdir -p $MODEL_PATH
# fi
# if [ ! -e $ENCODER ]; then
#   wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
# fi
# if [ ! -e $DECODER ]; then
#   wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
# fi

if [ -z "$DOWNLOAD_ONLY" ]
then

# Inference
python3 -u test_multipro.py \
  --data_dir $1 \
  --out_dir $2 \
  --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml \
  --gpus $3

fi
