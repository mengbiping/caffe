#!/bin/bash

CAFFE_MODEL=/mnt/data/models/clothes_googlenet/quick_snapshots/clothes_googlenet_iter_98000.caffemodel
MODEL_PROTO=models/clothes_googlenet/deploy.prototxt
MEAN_FILE=/mnt/data/data/clothes/clothes_mean.binaryproto
LABEL_FILE=data/clothes/label.txt
IMAGE_LIST_FILE=data/clothes/test.txt
PREDICATION_FILE=output.txt
# 0 for gpu, 1 for cpu
MODE=0

build/tools/classification \
    --logtostderr \
    --caffe_model="$CAFFE_MODEL" \
    --model_proto="$MODEL_PROTO" \
    --mean_file="$MEAN_FILE" \
    --mode=$MODE \
    --label_file="$LABEL_FILE" \
    --images_list_file="$IMAGE_LIST_FILE" \
    --output_predication="$PREDICATION_FILE"
