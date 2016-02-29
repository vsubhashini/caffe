#!/usr/bin/env sh
# This script downloads the trained S2VT VGG (RGB) model,
# associated vocabulary, and frame features for the validation set.

echo "Downloading Model and Data [~400MB] ..."

wget --no-check-certificate https://www.dropbox.com/s/wn6k2oqurxzt6e2/s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel
wget --no-check-certificate https://www.dropbox.com/s/20mxirwrqy1av01/yt_allframes_vgg_fc7_val.txt
wget --no-check-certificate https://www.dropbox.com/s/v1lrc6leknzgn3x/yt_coco_mvad_mpiimd_vocabulary.txt

echo "Organizing..."

DIR="./snapshots"
if [ ! -d "$DIR" ]; then
    mkdir $DIR
fi
mv s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel $DIR"/s2vt_vgg_rgb.caffemodel"

echo "Done."
