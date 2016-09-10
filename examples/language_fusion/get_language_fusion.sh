#!/usr/bin/env sh
# This script downloads the trained Deep Fusion (In Domain) model,
# associated vocabulary, and frame features for the validation set.

echo "Downloading Model and Data [~1GB] ..."

wget --no-check-certificate https://www.dropbox.com/s/7g9qt4bt1p7q5bw/lm_deepfus_img512_s2vt_glove_72.7kvocab_sgd_lr2e3step7k_iter_16000.caffemodel
wget --no-check-certificate https://www.dropbox.com/s/20mxirwrqy1av01/yt_allframes_vgg_fc7_val.txt
wget --no-check-certificate https://www.dropbox.com/s/mypl5nxrzemzdm2/vocabulary_72k_surf_intersect_glove.txt

echo "Organizing..."

DIR="./snapshots"
if [ ! -d "$DIR" ]; then
    mkdir $DIR
fi
mv lm_deepfus_img512_s2vt_glove_72.7kvocab_sgd_lr2e3step7k_iter_16000.caffemodel $DIR"/indomain_deepfusion.caffemodel"

echo "Done."
