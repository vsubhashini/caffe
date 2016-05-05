## S2VT: Sequence to Sequence Video to Text ##

To train the S2VT model you will need to compile from my recurrent branch of caffe:
```
    git clone https://github.com/vsubhashini/caffe.git
    git checkout recurrent
```
To compile Caffe, please refer to the [Installation page](http://caffe.berkeleyvision.org/installation.html).

### Using the model to generate captions

**Get preprocessed model and sample data**
```
    ./get_s2vt.sh
```
**Run the captioner**
```
    python s2vt_captioner.py -m s2vt_vgg_rgb
```
### Preparing data for videos

1. **Pre-process videos to get frame features.** The code provided here does
not process videos directly. You can use any method to sample video frames and
extract VGG features for the frames. You might want the features to be
formatted similar to the sample data in the download script. The sample data
corresponds to the validation set of the Youtube Dataset.

2. **Convert features to hdf5.** If your features are in text format use
`framefc7_stream_text_to_hdf5_data.py` to convert to hdf5 data. If they are in a
mat file you might want to use `framefc7_stream_mat_text_to_hdf5_data.py`.

### Training the model

1. **Point to the hdf5 training data.** Modify `s2vt.prototxt` to point to the
hdf5 training and validation data.
2. **Train the model.** Use `s2vt_solver.prototxt` to train your model.

### Evaluating the generated sentences.

Code to evaluate the predicted sentences (with example) can be found at
[https://github.com/vsubhashini/caption-eval](https://github.com/vsubhashini/caption-eval).

### Reference

If you find this code helpful, please consider citing:

[Sequence to Sequence - Video to Text](https://vsubhashini.github.io/s2vt.html)

    Sequence to Sequence - Video to Text
    S. Venugopalan, M. Rohrbach, J. Donahue, T. Darrell, R. Mooney, K. Saenko
    The IEEE International Conference on Computer Vision (ICCV) 2015

