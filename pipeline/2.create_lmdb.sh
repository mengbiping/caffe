#!/usr/bin/env sh
# Create the lmdb inputs for training/validation images.
# N.B. set the path to the train + val data dirs

DATA_NAME=clothes_neck
CAFFE_ROOT=$HOME/caffe

cd $CAFFE_ROOT
echo "Entering ${CAFFE_ROOT}"

TOOLS=build/tools
DATA_PATH=data/$DATA_NAME
TRAIN_DATA_ROOT=$DATA_PATH/train/
VAL_DATA_ROOT=$DATA_PATH/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# RESIZE=false
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA_PATH/train.txt \
    $DATA_PATH/${DATA_NAME}_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA_PATH/val.txt \
    $DATA_PATH/${DATA_NAME}_val_lmdb

echo "Done."
