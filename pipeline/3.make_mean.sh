#!/usr/bin/env sh
# Compute the mean image from the training lmdb.

OUTPUT_DIR=.
if [ $# -ge 1 ]; then
    OUTPUT_DIR="$1"
fi

DATA_NAME=clothes
if [ $# -ge 2 ]; then
    DATA_NAME="$2"
fi

CAFFE_ROOT=$HOME/caffe

cd $CAFFE_ROOT
echo "Entering ${CAFFE_ROOT}"

TOOLS=build/tools
DATA_PATH=$OUTPUT_DIR/data/$DATA_NAME

$TOOLS/compute_image_mean $DATA_PATH/${DATA_NAME}_train_lmdb \
  $DATA_PATH/${DATA_NAME}_mean.binaryproto

echo "Done."
