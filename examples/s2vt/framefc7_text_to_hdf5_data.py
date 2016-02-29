#!/usr/bin/env python
# fc7 for every frame in text to hdf

import csv
import numpy as np
import os
import random
random.seed(3)
import sys

from hdf5_npsequence_generator import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<en_unk>'

# start every sentence in a new array, pad if <max
MAX_WORDS = 80

"""Filenames has file with vgg fc7 frame feats for vidids
   and sentences with video ids"""
class fc7FrameSequenceGenerator(SequenceGenerator):
  def __init__(self, filenames, batch_num_streams=1, vocab_filename=None,
               max_words=MAX_WORDS, align=True, shuffle=True, pad=True,
               truncate=True, reverse=False):
    self.max_words = max_words
    self.reverse = reverse
    self.lines = []
    num_empty_lines = 0
    self.vid_framefeats = {} # listofdict [{}]
    for framefeatfile, sentfile in filenames:
      print 'Reading frame features from file: %s' % framefeatfile
      with open(framefeatfile, 'rb') as featfd:
        # each line has the fc7 for 1 frame in video
        pool_csv = csv.reader(featfd)
        pool_csv = list(pool_csv)
        for line in pool_csv:
          id_framenum = line[0]
          video_id = id_framenum.split('_')[0]
          if video_id not in self.vid_framefeats:
            self.vid_framefeats[video_id]=[]
          self.vid_framefeats[video_id].append(','.join(line[1:]))
      if sentfile:
        print 'Reading sentences in: %s' % sentfile
        with open(sentfile, 'r') as sentfd:
          for line in sentfd:
            line = line.strip()
            id_sent = line.split('\t')
            if len(id_sent)<2:
              num_empty_lines += 1
              continue
            self.lines.append((id_sent[0], id_sent[1]))
      if num_empty_lines > 0:
        print 'Warning: ignoring %d empty lines.' % num_empty_lines
    self.line_index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.frame_list = []
    self.vocabulary = {}
    self.vocabulary_inverted = []
    self.vocab_counts = []
    # initialize vocabulary
    self.init_vocabulary(vocab_filename)
    SequenceGenerator.__init__(self)
    self.batch_num_streams = batch_num_streams  # needed in hdf5 to seq
    # make the number of image/sentence pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    if align:
      num_pairs = len(self.lines)
      remainder = num_pairs % BUFFER_SIZE
      if remainder > 0:
        num_needed = BUFFER_SIZE - remainder
        for i in range(num_needed):
          choice = random.randint(0, num_pairs - 1)
          self.lines.append(self.lines[choice])
      assert len(self.lines) % BUFFER_SIZE == 0
    if shuffle:
      random.shuffle(self.lines)
    self.pad = pad
    self.truncate = truncate
    self.negative_one_padded_streams = frozenset(('target_sentence'))

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename):
    print "Initializing the vocabulary."
    if os.path.isfile(vocab_filename):
      with open(vocab_filename, 'rb') as vocab_file:
        self.init_vocab_from_file(vocab_file)
    else:
      self.init_vocabulary_from_data(vocab_filename)

  def init_vocab_from_file(self, vocab_filedes):
    # initialize the vocabulary with the UNK word
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted = [UNK_IDENTIFIER]
    num_words_dataset = 0
    for line in vocab_filedes.readlines():
      split_line = line.split()
      word = split_line[0]
      print word
      if word == UNK_IDENTIFIER:
        continue
      else:
        assert word not in self.vocabulary
      num_words_dataset += 1
      self.vocabulary[word] = len(self.vocabulary_inverted)
      self.vocabulary_inverted.append(word)
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized vocabulary from file with %d unique words ' +
           '(from %d total words in dataset).') % \
          (num_words_vocab, num_words_dataset)
    assert len(self.vocabulary_inverted) == num_words_vocab

  def init_vocabulary_from_data(self, vocab_filename):
    print 'Initializing the vocabulary from full data'
    assert len(self.lines) > 0
    # initialize the vocabulary with the UNK word if new
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted.append(UNK_IDENTIFIER)
    # count frequency of word in data
    self.vocab_counts.append(0)
      
    num_words_dataset = 0
    for vidid, line in self.lines:
      split_line = line.split()
      num_words_dataset += len(split_line)
      for word in split_line:
        if word in self.vocabulary:
          self.vocab_counts[self.vocabulary[word]] += 1
        else:
          self.vocabulary_inverted.append(word)
          self.vocabulary[word] = len(self.vocab_counts)
          self.vocab_counts.append(1)
          
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized the vocabulary from data with %d unique words ' +
           '(from %d total words in dataset).') % (num_words_vocab, num_words_dataset)
    assert len(self.vocab_counts) == num_words_vocab
    assert len(self.vocabulary_inverted) == num_words_vocab
    if self.vocab_counts[self.vocabulary[UNK_IDENTIFIER]] == 0:
      print 'Warning: the count for the UNK identifier "%s" was 0.' % UNK_IDENTIFIER

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def dump_video_file(self, vidid_order_file, frame_seq_label_file):
    print 'Dumping vidid order to file: %s' % vidid_order_file
    with open(vidid_order_file,'wb') as vidid_file:
      for vidid, line in self.lines:
        word_count = len(line.split())
        frame_count = len(self.vid_framefeats[vidid])
        total_count = word_count +frame_count
        vidid_file.write('%s\t%d\t%d\t%d\n' % (vidid, word_count, frame_count, total_count))
    print 'Done.' 

  def next_line(self):
    num_lines = float(len(self.lines))
    if self.line_index == 1 or self.line_index == num_lines or self.line_index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.line_index, num_lines,
                                              100 * self.line_index / num_lines)
    self.line_index += 1
    if self.line_index == num_lines:
      self.line_index = 0
      self.num_resets += 1

  def get_pad_value(self, stream_name):
    return -1 if stream_name in self.negative_one_padded_streams else 0


  def float_line_to_stream(self, line):
    return map(float, line.split(','))

  def line_to_stream(self, sentence):
    stream = []
    for word in sentence.split():
      word = word.strip()
      if word in self.vocabulary:
        stream.append(self.vocabulary[word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[UNK_IDENTIFIER])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  # we have pooled fc7 features already in the file
  def get_streams(self):
    vidid, line = self.lines[self.line_index]
    assert vidid in self.vid_framefeats
    feats_vgg_fc7 = self.vid_framefeats[vidid] # list of fc7 feats for all frames
    num_frames = len(feats_vgg_fc7)
    stream = self.line_to_stream(line)
    num_words = len(stream)
    pad = self.max_words - (num_words + 1 + num_frames) if self.pad else 0
    truncated = False
    if pad < 0:
      # print '{0} #frames: {1} #words: {2}'.format(vidid, num_frames, num_words)
      # truncate frames
      if (num_words + 1) > self.max_words:
        stream = stream[:20] # truncate words to 20
      num_frames = self.max_words - (len(stream)+1)
      truncated = True
      pad = 0
      # print 'Truncated_{0}: #frames: {1} #words: {2}'.format(vidid, num_frames, len(stream))
      self.num_truncates += truncated

    # reverse the string
    if self.reverse:
      stream.reverse()

    if pad > 0: self.num_pads += 1
    self.num_outs += 1
    out = {}
    out['cont_sentence'] = [0] + [1] * (num_frames +len(stream)) + [0] * pad
    out['input_sentence'] = [0] * num_frames + [0] + stream + [0] * pad
    out['target_sentence'] = [-1] * num_frames + stream + [0] + [-1] * pad
    # For encoder-decoder model
    out['cont_img'] = [0] + [1] * (num_frames - 1) + [0] * (len(stream) + 1 + pad)
    out['cont_sen'] = [0] * (num_frames + 1) + [1] * len(stream) + [0] * pad
    out['encoder_to_decoder'] = [0] * (num_frames - 1) + [1] + [0] * (len(stream) + 1 + pad)
    out['stage_indicator'] = [0] * num_frames + [1] * (len(stream) + 1 + pad)
    out['inv_stage_indicator'] = [1] * num_frames + [0] * (len(stream) + 1 + pad)
    # fc7 features
    out['frame_fc7'] = []
    for frame_feat in feats_vgg_fc7[:num_frames]:
      feat_fc7 = self.float_line_to_stream(frame_feat)
      out['frame_fc7'].append(np.array(feat_fc7).reshape(1, len(feat_fc7)))
    # pad last frame for the length of the sentence
    num_img_pads = len(out['input_sentence']) - num_frames
    zero_padding = np.zeros(len(feat_fc7)).reshape(1, len(feat_fc7))
    for padframe in range(num_img_pads):
      out['frame_fc7'].append(zero_padding)
    assert len(out['frame_fc7'])==len(out['input_sentence'])
    self.next_line()
    return out


BUFFER_SIZE = 1 # TEXT streams
BATCH_STREAM_LENGTH = 1000
SETTING = '.'
OUTPUT_DIR = '{0}/hdf5/buffer_{1}_s2vt_{2}'.format(SETTING, BUFFER_SIZE, MAX_WORDS)
VOCAB = '%s/vocabulary.txt' % SETTING
OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
FRAMEFEAT_FILE_PATTERN = './youtube/splits/yt_allframes_vgg_fc7_{0}.txt'
SENTS_FILE_PATTERN = './youtube/splits/sents_{0}_lc_nopunc.txt'
OUT_FILE_PATTERN = \
'./rawcorpus/{0}/s2vt_vgg_{0}_sequence.txt'
OUT_CORPUS_PATH = './rawcorpus/{0}'

def preprocess_dataset(split_name, data_split_name, batch_stream_length,
                      aligned=False, reverse=False):
  filenames = [(FRAMEFEAT_FILE_PATTERN.format(data_split_name),
               SENTS_FILE_PATTERN.format(data_split_name))]
  vocab_filename = VOCAB
  output_path = OUTPUT_DIR_PATTERN % split_name
  aligned = True
  fsg = fc7FrameSequenceGenerator(filenames, BUFFER_SIZE, vocab_filename,
         max_words=MAX_WORDS, align=aligned, shuffle=True, pad=aligned,
         truncate=aligned, reverse=reverse)
  fsg.batch_stream_length = batch_stream_length
  writer = HDF5SequenceWriter(fsg, output_dir=output_path)
  writer.write_to_exhaustion()
  writer.write_filelists()
  if not os.path.isfile(vocab_filename):
    print "Vocabulary not found"
    # fsg.dump_vocabulary(vocab_out_path)
  out_path = OUT_CORPUS_PATH.format(data_split_name)
  vid_id_order_outpath = '%s/yt_s2vtvgg_%s_vidid_order_%d_%d.txt' % \
  (out_path, data_split_name, BUFFER_SIZE, MAX_WORDS)
  frame_sequence_outpath = '%s/yt_s2vtvgg_%s_sequence_%d_%d_recurrent.txt' % \
  (out_path, data_split_name, BUFFER_SIZE, MAX_WORDS)
  fsg.dump_video_file(vid_id_order_outpath, frame_sequence_outpath)

def process_splits():
  DATASETS = [ # split_name, data_split_name, aligned
      ('train', 'train', False, False),
      ('valid', 'val', False, False),
  ]
  for split_name, data_split_name, aligned, reverse in DATASETS:
    preprocess_dataset(split_name, data_split_name, BATCH_STREAM_LENGTH,aligned,
    reverse)

if __name__ == "__main__":
  process_splits()
