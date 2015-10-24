#!/usr/bin/python
# Create a npy file from the binaryproto image mean file.

import argparse
import numpy as np
import os
import sys

caffe_path = os.path.join(os.getenv("HOME"), "caffe", "python")
sys.path.insert(0, caffe_path)
import caffe

parser = argparse.ArgumentParser()
parser.add_argument("--data_path_prefix", default="/mnt/data")
parser.add_argument("--data_name", default="clothes")
parser.add_argument("--input_suffix", default="_mean.binaryproto")
parser.add_argument("--output_suffix", default="_mean.npy")
args = parser.parse_args()

if len(args.data_name) <= 0:
    print "Usage: python convert_protomean.py --data_name clothes"
    sys.exit()

# print sys.argv[0]
input_file = os.path.join(args.data_path_prefix, "data", args.data_name,
        "%s%s" % (args.data_name, args.input_suffix))
output_file = os.path.join(args.data_path_prefix, "data", args.data_name,
        "%s%s" % (args.data_name, args.output_suffix))
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(input_file, 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
print 'Length:', len(arr)
out = arr[0]
np.save(output_file, out )
print "%s created." % output_file
