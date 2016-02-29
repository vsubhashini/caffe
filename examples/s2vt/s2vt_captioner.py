DEVICE_ID = 0

""" Script to generate captions from video features"""

from collections import OrderedDict
import argparse
import cPickle as pickle
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

sys.path.append('../../python/')
import caffe

from framefc7_text_to_hdf5_data import *

def vocab_inds_to_sentence(vocab, inds):
  sentence = ' '.join([vocab[i] for i in inds])
  # Capitalize first character.
  sentence = sentence[0].upper() + sentence[1:]
  # Replace <EOS> with '.', or append '...'.
  if sentence.endswith(' ' + vocab[0]):
    sentence = sentence[:-(len(vocab[0]) + 1)] + '.'
  else:
    sentence += '...'
  return sentence
  
def video_to_descriptor(video_id, fsg):
  video_features = []
  assert video_id in fsg.vid_framefeats
  all_frames_fc7 = fsg.vid_framefeats[video_id]
  for frame_fc7 in all_frames_fc7:
    frame_fc7 = fsg.float_line_to_stream(frame_fc7)
    video_features.append(np.array(frame_fc7).reshape(1, len(frame_fc7)))
  return video_features

def encode_video_frames(net, video_features, previous_word=-1):
  for frame_feature in video_features:
    cont_input = 0 if previous_word == -1 else 1
    previous_word = 0
    cont = np.array([cont_input])
    data_en = np.array([previous_word])
    stage_ind = np.array([0]) # encoding stage
    image_features = np.zeros_like(net.blobs['frames_fc7'].data)
    image_features[:] = frame_feature
    net.forward(frames_fc7=image_features, cont_sentence=cont, input_sentence=data_en,
       stage_indicator=stage_ind)

def predict_single_word(net, pad_img_feature, previous_word, output='probs'):
  cont_input = 1
  cont = np.array([cont_input])
  data_en = np.array([previous_word])
  stage_ind = np.array([1]) # decoding stage
  image_features = np.zeros_like(net.blobs['frames_fc7'].data)
  image_features[:] = pad_img_feature
  net.forward(frames_fc7=image_features, cont_sentence=cont, input_sentence=data_en,
     stage_indicator=stage_ind)
  output_preds = net.blobs[output].data.reshape(-1)
  return output_preds

def predict_single_word_from_all_previous(net, pad_img_feature, previous_words):
  probs = predict_single_word(net, pad_img_feature, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, pad_img_feature, word)
  return probs

# Strategy must be either 'beam' or 'sample'.
# If 'beam', do a max likelihood beam search with beam size num_samples.
# Otherwise, sample with temperature temp.
def predict_image_caption(net, pad_img_feature, vocab_list, strategy={'type': 'beam'}):
  assert 'type' in strategy
  assert strategy['type'] in ('beam', 'sample')
  if strategy['type'] == 'beam':
    return predict_image_caption_beam_search(net, pad_img_feature, vocab_list, strategy)
  num_samples = strategy['num'] if 'num' in strategy else 1
  samples = []
  sample_probs = []
  for _ in range(num_samples):
    sample, sample_prob = sample_image_caption(net, pad_img_feature, strategy)
    samples.append(sample)
    sample_probs.append(sample_prob)
  return samples, sample_probs

def softmax(softmax_inputs, temp):
  exp_inputs = np.exp(temp * softmax_inputs)
  exp_inputs_sum = exp_inputs.sum()
  if math.isnan(exp_inputs_sum):
    return exp_inputs * float('nan')
  elif math.isinf(exp_inputs_sum):
    assert exp_inputs_sum > 0  # should not be -inf
    return np.zeros_like(exp_inputs)
  eps_sum = 1e-8
  return exp_inputs / max(exp_inputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1.0, already_softmaxed=False):
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1.0
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def sample_image_caption(net, image, strategy, net_output='predict', max_length=50):
  sentence = []
  probs = []
  eps_prob = 1e-8
  temp = strategy['temp'] if 'temp' in strategy else 1.0
  if max_length < 0: max_length = float('inf')
  while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
    previous_word = sentence[-1] if sentence else 0
    softmax_inputs = \
        predict_single_word(net, image, previous_word, output=net_output)
    word = random_choice_from_probs(softmax_inputs, temp)
    sentence.append(word)
    probs.append(softmax(softmax_inputs, 1.0)[word])
  return sentence, probs

def predict_image_caption_beam_search(net, pad_img_feature, vocab_list, strategy, max_length=50):
  # Note: This code support S2VT only for beam-width 1.
  beam_size = 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if len(beam) >= max_length or previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          # Don't expand this beam; it was already ended with an EOS,
          # or is the max length.
          continue
      else:
        previous_word = 0  # EOS is first word
      if beam_size == 1:
        probs = predict_single_word(net, pad_img_feature, previous_word)
      else:
        probs = predict_single_word_from_all_previous(net, pad_img_feature, beam)
      assert len(probs.shape) == 1
      assert probs.shape[0] == len(vocab_list)
      expansion_inds = probs.argsort()[-beam_size:]
      for ind in expansion_inds:
        prob = probs[ind]
        extended_beam_log_prob = beam_log_prob + math.log(prob)
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probabilitf.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    new_beam_probs = \
        [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
    beam_log_probs = [e['log_prob'] for e in expansions]
    beams_complete = 0
    for beam in new_beams:
      if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
    beams, beam_probs = new_beams, new_beam_probs
  return beams, beam_probs

def run_pred_iter(net, pad_image_feature, vocab_list, strategies=[{'type': 'beam'}]):
  outputs = []
  for strategy in strategies:
    captions, probs = predict_image_caption(net, pad_image_feature, vocab_list, strategy=strategy)
    for caption, prob in zip(captions, probs):
      output = {}
      output['caption'] = caption
      output['prob'] = prob
      output['gt'] = False
      output['source'] = strategy
      outputs.append(output)
  return outputs

def score_caption(net, image, caption, is_gt=True, caption_source='gt'):
  output = {}
  output['caption'] = caption
  output['gt'] = is_gt
  output['source'] = caption_source
  output['prob'] = []
  probs = predict_single_word(net, image, 0)
  for word in caption:
    output['prob'].append(probs[word])
    probs = predict_single_word(net, image, word)
  return output

def next_video_gt_pair(tsg):
  streams = tsg.get_streams()
  video_id = tsg.lines[tsg.line_index-1][0]
  gt = streams['target_sentence']
  return video_id, gt

def all_video_gt_pairs(fsg):
  data = OrderedDict()
  if len(fsg.lines) > 0:
    prev_video_id = None
    while True:
      video_id, gt = next_video_gt_pair(fsg)
      if video_id in data:
        if video_id != prev_video_id:
          break
        data[video_id].append(gt)
      else:
        data[video_id] = [gt]
      prev_video_id = video_id
    print 'Found %d videos with %d captions' % (len(data.keys()), len(data.values()))
  else:
    data = OrderedDict(((key, []) for key in fsg.vid_framefeats.keys()))
  return data

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = \
        stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats

def run_pred_iters(pred_net, vidids, video_gt_pairs, fsg,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = OrderedDict()
  num_pairs = 0
  descriptor_video_id = ''
  pad_img_feature = None
  for video_id in vidids:
    gt_captions = video_gt_pairs[video_id] # gets the target stream
    assert video_id not in outputs
    num_pairs += 1
    if descriptor_video_id != video_id:
      # get fc7 feature for the video
      video_features = video_to_descriptor(video_id, fsg)
      print 'Num video features: %d ' % len(video_features)
      print 'Dimension of video features: {0}'.format(video_features[0].shape)
      # run lstm on all the frames of video before predicting
      encode_video_frames(pred_net, video_features)
      # use the last frame from the video as padding
      pad_img_feature = video_features[-1]
      # Make padding all 0 when predicting
      pad_img_feature[pad_img_feature > 0] = 0
      desciptor_video_id = video_id
    outputs[video_id] = \
        run_pred_iter(pred_net, pad_img_feature, display_vocab, strategies=strategies)
    # for gt_caption in gt_captions:
    #   outputs[image_path].append(
    #       score_caption(pred_net, pad_img_feature, gt_caption))
    if display_vocab is not None:
      for output in outputs[video_id]:
        caption, prob, gt, source = \
            output['caption'], output['prob'], output['gt'], output['source']
        caption_string = vocab_inds_to_sentence(display_vocab, caption)
        if gt:
          tag = 'Actual'
        else:
          tag = 'Generated'
        stats = gen_stats(prob)
        print '%s caption (length %d, log_p = %f, log_p_word = %f):\n%s' % \
            (tag, stats['length'], stats['log_p'], stats['log_p_word'], caption_string)
  return outputs

def to_html_row(columns, header=False):
  out= '<tr>'
  for column in columns:
    if header: out += '<th>'
    else: out += '<td>'
    try:
      if int(column) < 1e8 and int(column) == float(column):
        out += '%d' % column
      else:
        out += '%0.04f' % column
    except:
      out += '%s' % column
    if header: out += '</th>'
    else: out += '</td>'
  out += '</tr>'
  return out

def to_html_output(outputs, vocab):
  out = ''
  for video_id, captions in outputs.iteritems():
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    out += '<img src="%s"><br>\n' % video_id
    out += '<table border="1">\n'
    column_names = ('Source', '#Words', 'Perplexity/Word', 'Caption')
    out += '%s\n' % to_html_row(column_names, header=True)
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if gt:
        source = 'ground truth'
        if 'correct' in c:
          caption_string = '<font color="%s">%s</font>' % \
              ('green' if c['correct'] else 'red', caption_string)
        else:
          caption_string = '<em>%s</em>' % caption_string
      else:
        if source['type'] == 'beam':
          source = 'beam (size %d)' % source['beam_size']
        elif source['type'] == 'sample':
          source = 'sample (temp %f)' % source['temp']
        else:
          raise Exception('Unknown type: %s' % source['type'])
        caption_string = '<strong>%s</strong>' % caption_string
      columns = (source, stats['length'] - 1,
                 stats['perplex_word'], caption_string)
      out += '%s\n' % to_html_row(columns)
    out += '</table>\n'
    out += '<br>\n\n' 
    out += '<br>' * 2
  out.replace('<unk>', 'UNK')  # sanitize...
  return out

def to_text_output(outputs, vocab):
  out_types = {}
  caps = outputs[outputs.keys()[0]]
  for c in caps:
    caption, gt, source = \
        c['caption'], c['gt'], c['source']
    if source['type'] == 'beam':
      source_meta = 'beam_size_%d' % source['beam_size']
    elif source['type'] == 'sample':
      source_meta = 'sample_temp_ %f' % source['temp']
    else:
      raise Exception('Unknown type: %s' % source['type'])
    if source_meta not in out_types:
      out_types[source_meta] = []
  num_videos = 0
  out = ''
  for video_id, captions in outputs.iteritems():
    num_videos += 1
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      source_meta = 'beam_size_%d' % source['beam_size']
      out = '%s\t%s\t%s\n' % (source_meta, video_id,caption_string)
      # if len(out_types[source_meta]) < num_videos:
      out_types[source_meta].append(out)
  return out_types

def retrieval_image_list(dataset, cache_dir):
  image_list_filename = '%s/image_paths.txt' % cache_dir
  if os.path.exists(image_list_filename):
    with open(image_list_filename, 'r') as image_list_file:
      image_paths = [i.strip() for i in image_list_file.readlines()]
      assert set(image_paths) == set(dataset.keys())
  else:
    image_paths = dataset.keys()
    with open(image_list_filename, 'w') as image_list_file:
      image_list_file.write('\n'.join(image_paths) + '\n')
  return image_paths

def compute_descriptors(net, image_list, output_name='fc7'):
  batch = np.zeros_like(net.blobs['data'].data)
  batch_shape = batch.shape
  batch_size = batch_shape[0]
  descriptors_shape = (len(image_list), ) + net.blobs[output_name].data.shape[1:]
  descriptors = np.zeros(descriptors_shape)
  for batch_start_index in range(0, len(image_list), batch_size):
    batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
    for batch_index, image_path in enumerate(batch_list):
      batch[batch_index:(batch_index + 1)] = preprocess_image(net, image_path)
    print 'Computing descriptors for images %d-%d of %d' % \
        (batch_start_index, batch_start_index + batch_size - 1, len(image_list))
    net.forward(data=batch)
    print 'Done'
    descriptors[batch_start_index:(batch_start_index + batch_size)] = \
        net.blobs[output_name].data
  return descriptors

def retrieval_descriptors(net, image_list, cache_dir):
  descriptor_filename = '%s/descriptors.npz' % cache_dir
  if os.path.exists(descriptor_filename):
    descriptors = np.load(descriptor_filename)['descriptors']
  else:
    descriptors = compute_descriptors(net, image_list)
    np.savez_compressed(descriptor_filename, descriptors=descriptors)
  return descriptors

def retrieval_caption_list(dataset, image_list, cache_dir):
  caption_list_filename = '%s/captions.pkl' % cache_dir
  if os.path.exists(caption_list_filename):
    with open(caption_list_filename, 'rb') as caption_list_file:
      captions = pickle.load(caption_list_file)
  else:
    captions = []
    for image in image_list:
      for caption in dataset[image]:
        captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    captions.sort(key=lambda c: len(c['caption']))
    with open(caption_list_filename, 'wb') as caption_list_file:
      pickle.dump(captions, caption_list_file)
  return captions

def sample_captions(net, image_features,
    prob_output_name='probs', output_name='samples', caption_source='sample'):
  cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
  word_input = np.zeros_like(net.blobs['input_sentence'].data)
  batch_size = image_features.shape[0]
  outputs = []
  output_captions = [[] for b in range(batch_size)]
  output_probs = [[] for b in range(batch_size)]
  caption_index = 0
  num_done = 0
  while num_done < batch_size:
    if caption_index == 0:
      cont_input[:] = 0
    elif caption_index == 1:
      cont_input[:] = 1
    if caption_index == 0:
      word_input[:] = 0
    else:
      for index in range(batch_size):
        word_input[index] = \
            output_captions[index][caption_index - 1] if \
            caption_index <= len(output_captions[index]) else 0
    net.forward(image_features=image_features,
        cont_sentence=cont_input, input_sentence=word_input)
    net_output_samples = net.blobs[output_name].data
    net_output_probs = net.blobs[prob_output_name].data
    for index in range(batch_size):
      # If the caption is empty, or non-empty but the last word isn't EOS,
      # predict another word.
      if not output_captions[index] or output_captions[index][-1] != 0:
        next_word_sample = net_output_samples[index]
        assert next_word_sample == int(next_word_sample)
        next_word_sample = int(next_word_sample)
        output_captions[index].append(next_word_sample)
        output_probs[index].append(net_output_probs[index, next_word_sample])
        if next_word_sample == 0: num_done += 1
    print '%d/%d done after word %d' % (num_done, batch_size, caption_index)
    caption_index += 1
  for prob, caption in zip(output_probs, output_captions):
    output = {}
    output['caption'] = caption
    output['prob'] = prob
    output['gt'] = False
    output['source'] = caption_source
    outputs.append(output)
  return outputs

def print_top_samples(vocab, samples, out_filename=None):
  top_sample = OrderedDict()
  for sample in samples:
    stats = gen_stats(sample['prob'])
    image_path = sample['source']
    if image_path not in top_sample:
      top_sample[image_path] = (None, -float('inf'))
    if stats['log_p_word'] > top_sample[image_path][1]:
      top_sample[image_path] = (sample['caption'], stats['log_p_word'])
  out_file = open(out_filename, 'w') if out_filename is not None else sys.stdout
  for image_path, sample in top_sample.iteritems():
    image_id = os.path.split(image_path)[1]
    out_file.write("%s\t%s\n" % (image_id, vocab_inds_to_sentence(vocab, sample[0])))
  out_file.close()
  print 'Wrote top samples to:', out_filename

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelname", type=str, required=True,
                    help='Name of model without ".caffemodel" extension')
  parser.add_argument("-t", "--testset", action='store_true',
                    help='Evaluate on test set. If unspecified then val set.')
  parser.add_argument("-o", "--htmlout", action='store_true',
                    help='output sentences as html to visually compare')
  parser.add_argument("-g", "--gold", action='store_true',
                    help='groundtruth sentences for scoring/retrieval')
  args = parser.parse_args()

  # TODO: Input the snapshot directory, vocab path, frames (and sents) path
  DIR = './snapshots'
  VOCAB_FILE = './yt_coco_mvad_mpiimd_vocabulary.txt'
  FRAMEFEAT_FILE_PATTERN = './yt_allframes_vgg_fc7_{0}.txt'
 
  LSTM_NET_FILE = './s2vt.words_to_preds.deploy.prototxt'
  RESULTS_DIR = './results'
  MODEL_FILE = '%s/%s.caffemodel' % (DIR, args.modelname)
  SENTS_FILE = args.gold if args.gold else None # optional
  NET_TAG = args.modelname
  if DEVICE_ID >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(DEVICE_ID)
  else:
    caffe.set_mode_cpu()
  print "Setting up LSTM NET"
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE, caffe.TEST)
  print "Done"
  nets = [lstm_net]

  STRATEGIES = [
    {'type': 'beam', 'beam_size': 1},
  ]
  NUM_OUT_PER_CHUNK = 30
  START_CHUNK = 0

  vocab_file = VOCAB_FILE
  DATASETS = [ ] # split_name, data_split_name, aligned
  if args.testset:
    DATASETS.append(('test', 'test', False))
  else:
    DATASETS.append(('valid', 'val', False))

  for split_name, data_split_name, aligned in DATASETS:
    filenames = [(FRAMEFEAT_FILE_PATTERN.format(data_split_name),
               SENTS_FILE)]
    fsg = fc7FrameSequenceGenerator(filenames, BUFFER_SIZE,
          vocab_file, max_words=MAX_WORDS, align=aligned, shuffle=False,
          pad=aligned, truncate=aligned)
    video_gt_pairs = all_video_gt_pairs(fsg)
    print 'Read %d videos pool feats' % len(fsg.vid_framefeats)
    NUM_CHUNKS = (len(fsg.vid_framefeats)/NUM_OUT_PER_CHUNK) + 1
    eos_string = '<EOS>'
    # add english inverted vocab 
    vocab_list = [eos_string] + fsg.vocabulary_inverted
    offset = 0
    for c in range(START_CHUNK, NUM_CHUNKS):
      chunk_start = c * NUM_OUT_PER_CHUNK
      chunk_end = (c + 1) * NUM_OUT_PER_CHUNK
      chunk = video_gt_pairs.keys()[chunk_start:chunk_end]
      html_out_filename = '%s/%s.%s.%d_to_%d.html' % \
          (RESULTS_DIR, data_split_name, NET_TAG, chunk_start, chunk_end)
      text_out_filename = '%s/%s.%s_' % \
          (RESULTS_DIR, data_split_name, NET_TAG)
      if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
      outputs = run_pred_iters(lstm_net, chunk, video_gt_pairs,
                    fsg, strategies=STRATEGIES, display_vocab=vocab_list)
      if args.htmlout:
        html_out = to_html_output(outputs, vocab_list)
        html_out_file = open(html_out_filename, 'w')
        html_out_file.write(html_out)
        html_out_file.close()
      text_out_types = to_text_output(outputs, vocab_list)
      for strat_type in text_out_types:
        text_out_fname = text_out_filename + strat_type + '.txt'
        text_out_file = open(text_out_fname, 'a')
        text_out_file.write(''.join(text_out_types[strat_type]))
        text_out_file.close()
      offset += NUM_OUT_PER_CHUNK
      print '(%d-%d) Appending to file: %s' % (chunk_start, chunk_end,
                                               text_out_fname)

if __name__ == "__main__":
 main()

