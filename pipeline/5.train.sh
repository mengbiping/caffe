#!/bin/bash
DATA_NAME=clothes_neck
NET_NAME=googlenet
CAFFE_ROOT=$HOME/caffe
cd $CAFFE_ROOT
echo "Entering ${CAFFE_ROOT}"

MODEL_NAME=${DATA_NAME}_${NET_NAME}
MODEL_PATH=models/$MODEL_NAME 
SOLVER=quick_solver

echo "Saving log to $MODEL_PATH/${SOLVER}.log"
build/tools/caffe train \
--solver=$MODEL_PATH/${SOLVER}.prototxt 2> $MODEL_PATH/${SOLVER}.log
