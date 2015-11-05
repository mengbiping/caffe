#!/usr/bin/env sh
# Create the lmdb inputs for training/validation images.
# N.B. set the path to the train + val data dirs

OUTPUT_DIR=.
if [ $# -ge 1 ]; then
    OUTPUT_DIR="$1"
fi
echo "OUTPUT_DIR: $OUTPUT_DIR"

DATA_NAME=clothes
if [ $# -ge 2 ]; then
    DATA_NAME="$2"
fi
echo "DATA_NAME: $DATA_NAME"
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

TRAIN_LMDB="$OUTPUT_DIR/$DATA_PATH/${DATA_NAME}_train_lmdb"
if [ -d "$TRAIN_LMDB" ]; then
    echo "Remove previous train lmdb $TRAIN_LMDB"
    rm -rf "$TRAIN_LMDB"
fi
echo "Create directory for train lmdb: $TRAIN_LMDB"
mkdir -p `dirname $TRAIN_LMDB`

VAL_LMDB="$OUTPUT_DIR/$DATA_PATH/${DATA_NAME}_val_lmdb"
if [ -d "$VAL_LMDB" ]; then
    echo "Remove previous val lmdb $VAL_LMDB"
    rm -rf "$VAL_LMDB"
fi
echo "Create directory for val lmdb: $VAL_LMDB"
mkdir -p `dirname $VAL_LMDB`

echo "Creating train lmdb in $TRAIN_LMDB ..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA_PATH/train.txt \
    $TRAIN_LMDB

echo "Creating val lmdb in $VAL_LMDB ..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA_PATH/val.txt \
    $VAL_LMDB

echo "Done."
