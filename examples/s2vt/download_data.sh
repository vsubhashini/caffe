#!/usr/bin/env sh
# This script downloads the video frame features for the training,
# validation, and test videos of the MSVD corpus.

echo "Downloading MSVD video features [~1.2GB] ..."

DIR="./data"
if [ ! -d "$DIR" ]; then
    mkdir $DIR
fi
if [ ! -f ./data/yt_allframes_vgg_fc7_val.txt ]; then
    echo "Downloading validation video features [~70MB] ..."
    wget --no-check-certificate https://www.dropbox.com/s/20mxirwrqy1av01/yt_allframes_vgg_fc7_val.txt
    mv yt_allframes_vgg_fc7_val.txt $DIR"/"
fi
if [ ! -f ./data/yt_allframes_vgg_fc7_test.txt ]; then
    echo "Downloading test video features [~440MB] ..."
    wget --no-check-certificate https://www.dropbox.com/s/n1857anlodhdkm0/yt_allframes_vgg_fc7_test.txt
    mv yt_allframes_vgg_fc7_test.txt $DIR"/"
fi
if [ ! -f ./data/yt_allframes_vgg_fc7_train.txt ]; then
    echo "Downloading test video features [~720MB] ..."
    wget --no-check-certificate https://www.dropbox.com/s/p2rszmjz0o0odnx/yt_allframes_vgg_fc7_train.txt
    mv yt_allframes_vgg_fc7_train.txt $DIR"/"
fi

echo "Done."
