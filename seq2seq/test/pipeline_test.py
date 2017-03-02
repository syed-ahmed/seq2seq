# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Cases for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import imp
import os
import shutil
import tempfile
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.test import utils as test_utils

BIN_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../bin"))

def _clear_flags():
  """Resets Tensorflow's FLAG values"""
  #pylint: disable=W0212
  tf.app.flags.FLAGS = tf.app.flags._FlagValues()
  tf.app.flags._global_parser = argparse.ArgumentParser()

class PipelineTest(tf.test.TestCase):
  """Tests training and inference scripts.
  """
  def setUp(self):
    super(PipelineTest, self).setUp()
    self.output_dir = tempfile.mkdtemp()
    self.bin_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../bin"))

  def tearDown(self):
    shutil.rmtree(self.output_dir, ignore_errors=True)
    super(PipelineTest, self).tearDown()

  def test_train_infer(self):
    """Tests training and inference scripts.
    """
    # Create dummy data
    sources_train, targets_train = test_utils.create_temp_parallel_data(
        sources=["a a a a", "b b b b", "c c c c", "笑 笑 笑 笑"],
        targets=["b b b b", "a a a a", "c c c c", "泣 泣 泣 泣"])
    sources_dev, targets_dev = test_utils.create_temp_parallel_data(
        sources=["a a", "b b", "c c c", "笑 笑 笑"],
        targets=["b b", "a a", "c c c", "泣 泣 泣"])
    vocab_source = test_utils.create_temporary_vocab_file(["a", "b", "c", "笑"])
    vocab_target = test_utils.create_temporary_vocab_file(["a", "b", "c", "泣"])

    _clear_flags()
    tf.reset_default_graph()
    train_script = imp.load_source(
        "seq2seq.test.train_bin", os.path.join(BIN_FOLDER, "train.py"))

    # Set training flags
    tf.app.flags.FLAGS.output_dir = self.output_dir
    tf.app.flags.FLAGS.metrics = """
      log_perplexity,bleu,rouge_1/f_score,rouge_l/f_score"""
    tf.app.flags.FLAGS.train_source = sources_train.name
    tf.app.flags.FLAGS.train_target = targets_train.name
    tf.app.flags.FLAGS.vocab_source = vocab_source.name
    tf.app.flags.FLAGS.vocab_target = vocab_target.name
    tf.app.flags.FLAGS.hparams = """{"embedding.dim": 10}"""
    tf.app.flags.FLAGS.model = "AttentionSeq2Seq"
    tf.app.flags.FLAGS.batch_size = 2

    # We pass a few flags via a config file
    config_path = os.path.join(self.output_dir, "train_config.yml")
    with gfile.GFile(config_path, "w") as config_file:
      yaml.dump({
          "dev_source":  sources_dev.name,
          "dev_target":  targets_dev.name,
          "train_steps": 50,
          "hparams": {
              "embedding.dim": 10,
              "attention.dim": 16,
              "decoder.rnn_cell.cell_class": "GRUCell",
              "decoder.rnn_cell.cell_params": {
                  "num_units": 8
              },
              "encoder.rnn_cell.cell_class": "GRUCell",
              "encoder.rnn_cell.cell_params": {
                  "num_units": 8
              }
          }}, config_file)

    tf.app.flags.FLAGS.config_path = config_path

    # Run training
    tf.logging.set_verbosity(tf.logging.INFO)
    train_script.main([])

    # Make sure a checkpoint was written
    expected_checkpoint = os.path.join(
        self.output_dir, "model.ckpt-50.data-00000-of-00001")
    self.assertTrue(os.path.exists(expected_checkpoint))

    # Reset flags and import inference script
    _clear_flags()
    tf.reset_default_graph()
    infer_script = imp.load_source(
        "seq2seq.test.infer_bin", os.path.join(BIN_FOLDER, "infer.py"))

    # Set inference flags
    attention_dir = os.path.join(self.output_dir, "att")
    tf.app.flags.FLAGS.model_dir = self.output_dir
    tf.app.flags.FLAGS.source = sources_dev.name
    tf.app.flags.FLAGS.batch_size = 2
    tf.app.flags.FLAGS.checkpoint_path = os.path.join(
        self.output_dir, "model.ckpt-50")
    tf.app.flags.FLAGS.dump_attention_dir = attention_dir

    # Make sure inference runs successfully
    infer_script.main([])

    # Make sure attention scores and visualizations exist
    self.assertTrue(os.path.exists(os.path.join(
        attention_dir, "attention_scores.npz")))
    self.assertTrue(os.path.exists(os.path.join(
        attention_dir, "00002.png")))

    # Load attention scores and assert shape
    scores = np.load(os.path.join(attention_dir, "attention_scores.npz"))
    self.assertIn("arr_0", scores)
    self.assertEqual(scores["arr_0"].shape[1], 3)
    self.assertIn("arr_1", scores)
    self.assertEqual(scores["arr_1"].shape[1], 3)
    self.assertIn("arr_2", scores)
    self.assertEqual(scores["arr_2"].shape[1], 4)
    self.assertIn("arr_3", scores)
    self.assertEqual(scores["arr_3"].shape[1], 4)

    # Test inference with beam search
    _clear_flags()
    tf.reset_default_graph()
    infer_script = imp.load_source(
        "seq2seq.test.infer_bin", os.path.join(BIN_FOLDER, "infer.py"))

    # Set inference flags
    tf.app.flags.FLAGS.model_dir = self.output_dir
    tf.app.flags.FLAGS.source = sources_dev.name
    tf.app.flags.FLAGS.batch_size = 2
    tf.app.flags.FLAGS.checkpoint_path = os.path.join(
        self.output_dir, "model.ckpt-50")
    tf.app.flags.FLAGS.hparams = """{
      "inference.beam_search.beam_width": 5
    }"""
    tf.app.flags.FLAGS.dump_beams = os.path.join(
        self.output_dir, "beams.npz")

    # Run inference w/ beam search
    infer_script.main([])
    self.assertTrue(os.path.exists(os.path.join(
        self.output_dir, "beams.npz")))



if __name__ == "__main__":
  tf.test.main()