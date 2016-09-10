## Improving LSTM-based Video Description with <br/> Linguistic Knowledge Mined from Text ##

An overview of the model is described at
[https://vsubhashini.github.io/language_fusion.html](https://vsubhashini.github.io/language_fusion.html)

To train the model you will need to compile from my recurrent branch of caffe:
```
    git clone https://github.com/vsubhashini/caffe.git
    git checkout recurrent
```
To compile Caffe, please refer to the [Installation page](http://caffe.berkeleyvision.org/installation.html).

### Using the model to generate captions

**Get preprocessed model and sample data**
```
    ./get_language_fusion.sh
```
**Run the captioner**
```
    python language_fusion_captioner.py -m indomain_deepfusion
```
### Preparing data for videos

Data preparation is identical to that of S2VT. Please refer to instructions
[here](https://github.com/vsubhashini/caffe/tree/recurrent/examples/s2vt).

1. **Pre-process videos to get frame features.** The code provided here does
not process videos directly. You can use any method to sample video frames and
extract VGG features for the frames. You might want the features to be
formatted similar to the sample data in the download script. The sample data
corresponds to the validation set of the Youtube Dataset.

2. **Convert features to hdf5.** If your features are in text format use
`framefc7_stream_text_to_hdf5_data.py` to convert to hdf5 data. If they are in a
mat file you might want to use `framefc7_stream_mat_text_to_hdf5_data.py`.

### Training the model


### Evaluating the generated sentences.

Code to evaluate the predicted sentences (with example) can be found at
[https://github.com/vsubhashini/caption-eval](https://github.com/vsubhashini/caption-eval).

### Reference

If you find this code helpful, please consider citing:

[Improving LSTM-based Video Description with Linguistic Knowledge Mined from
Text](https://vsubhashini.github.io/language_fusion.html)

    Improving LSTM-based Video Description with Linguistic Knowledge Mined from Text
    S. Venugopalan, L. Hendricks, R. Mooney, K. Saenko.
    Conference on Empirical Methods in Natural Language Processing (EMNLP) 2016

