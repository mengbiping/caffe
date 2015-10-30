#!/bin/bash

DATA_NAME=clothes
NET_NAME=googlenet
CAFFE_ROOT=$HOME/caffe
cd $CAFFE_ROOT
echo "Entering ${CAFFE_ROOT}"

MODEL_NAME=${DATA_NAME}_${NET_NAME}
MODEL_PATH=models/$MODEL_NAME 
SOLVER=quick_solver
SNAPSHOTS=`cat $MODEL_PATH/${SOLVER}.prototxt | grep "snapshot_prefix:" | sed "s/^snapshot_prefix: *\"//" | sed "s/\"$//"`
MAX_SNAPSHOT=`ls -t $SNAPSHOTS* | head | egrep "/clothes_googlenet_iter_[0-9]*.solverstate$" | head -1`

echo "MODEL_PATH: $MODEL_PATH"
echo "SNAPSHOTS: $SNAPSHOTS"
echo "MAX_SNAPSHOT: $MAX_SNAPSHOT"

echo "Saving log to $MODEL_PATH/${SOLVER}.log"

if [ ! -d $SNAPSHOTS ]; then
    mkdir -p $SNAPSHOTS
fi

if [ ! "$MAX_SNAPSHOT" == "" ] && [ -f $MAX_SNAPSHOT ]; then
  echo "Resuming from a snapshot: $MAX_SNAPSHOT"
  nohup build/tools/caffe train \
  --solver=$MODEL_PATH/${SOLVER}.prototxt --snapshot=$MAX_SNAPSHOT
  1>>$MODEL_PATH/${SOLVER}.log 2>>$MODEL_PATH/${SOLVER}.log &
else
  echo "Start training the model."
  nohup build/tools/caffe train \
  --solver=$MODEL_PATH/${SOLVER}.prototxt >$MODEL_PATH/${SOLVER}.log 2>&1 &
fi
