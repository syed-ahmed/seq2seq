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

"""Miscellaneous training utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import os
from collections import defaultdict
from pydoc import locate

import json
import random
import tensorflow as tf
from tensorflow import gfile

from seq2seq.contrib import rnn_cell
from tensorflow.python.ops.gen_image_ops import _adjust_contrastv2

slim = tf.contrib.slim


class TrainOptions(object):
    """A collection of options that are passed to the training script
    and can be saved to perform inference later.

    Args:
      task: Name of the training task class.
      task_params: A dictionary of parameters passed to the training task.
    """

    def __init__(self, model_class, model_params):
        self._model_class = model_class
        self._model_params = model_params

    @property
    def model_class(self):
        """Returns the training task parameters"""
        return self._model_class

    @property
    def model_params(self):
        """Returns the training task class"""
        return self._model_params

    @staticmethod
    def path(model_dir):
        """Returns the path to the options file.

        Args:
          model_dir: The model directory
        """
        return os.path.join(model_dir, "train_options.json")

    def dump(self, model_dir):
        """Dumps the options to a file in the model directory.

        Args:
          model_dir: Path to the model directory. The options will be
          dumped into a file in this directory.
        """
        gfile.MakeDirs(model_dir)
        options_dict = {
            "model_class": self.model_class,
            "model_params": self.model_params,
        }

        with gfile.GFile(TrainOptions.path(model_dir), "wb") as file:
            file.write(json.dumps(options_dict).encode("utf-8"))

    @staticmethod
    def load(model_dir):
        """ Loads options from the given model directory.

        Args:
          model_dir: Path to the model directory.
        """
        with gfile.GFile(TrainOptions.path(model_dir), "rb") as file:
            options_dict = json.loads(file.read().decode("utf-8"))
        options_dict = defaultdict(None, options_dict)

        return TrainOptions(
            model_class=options_dict["model_class"],
            model_params=options_dict["model_params"])


def cell_from_spec(cell_classname, cell_params):
    """Create a RNN Cell instance from a JSON string.

    Args:
      cell_classname: Name of the cell class, e.g. "BasicLSTMCell".
      cell_params: A dictionary of parameters to pass to the cell constructor.

    Returns:
      A RNNCell instance.
    """

    cell_params = cell_params.copy()

    # Find the cell class
    cell_class = getattr(rnn_cell, cell_classname)

    # Make sure additional arguments are valid
    cell_args = set(inspect.getargspec(cell_class.__init__).args[1:])
    for key in cell_params.keys():
        if key not in cell_args:
            raise ValueError(
                """{} is not a valid argument for {} class. Available arguments
                are: {}""".format(key, cell_class.__name__, cell_args))

    # Create cell
    return cell_class(**cell_params)


def get_rnn_cell(cell_class,
                 cell_params,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
    """Creates a new RNN Cell

    Args:
      cell_class: Name of the cell class, e.g. "BasicLSTMCell".
      cell_params: A dictionary of parameters to pass to the cell constructor.
      num_layers: Number of layers. The cell will be wrapped with
        `tf.contrib.rnn.MultiRNNCell`
      dropout_input_keep_prob: Dropout keep probability applied
        to the input of cell *at each layer*
      dropout_output_keep_prob: Dropout keep probability applied
        to the output of cell *at each layer*
      residual_connections: If true, add residual connections
        between all cells

    Returns:
      An instance of `tf.contrib.rnn.RNNCell`.
    """

    cells = []
    for _ in range(num_layers):
        cell = cell_from_spec(cell_class, cell_params)
        if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=dropout_input_keep_prob,
                output_keep_prob=dropout_output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = rnn_cell.ExtendedMultiRNNCell(
            cells=cells,
            residual_connections=residual_connections,
            residual_combiner=residual_combiner,
            residual_dense=residual_dense)
    else:
        final_cell = cells[0]

    return final_cell


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False):
    """Creates a function that decays the learning rate.

    Args:
      decay_steps: How often to apply decay.
      decay_rate: A Python number. The decay rate.
      start_decay_at: Don't decay before this step
      stop_decay_at: Don't decay after this step
      min_learning_rate: Don't decay below this number
      decay_type: A decay function name defined in `tf.train`
      staircase: Whether to apply decay in a discrete staircase,
        as opposed to continuous, fashion.

    Returns:
      A function that takes (learning_rate, global_step) as inputs
      and returns the learning rate for the given step.
      Returns `None` if decay_type is empty or None.
    """
    if decay_type is None or decay_type == "":
        return None

    start_decay_at = tf.to_int32(start_decay_at)
    stop_decay_at = tf.to_int32(stop_decay_at)

    def decay_fn(learning_rate, global_step):
        """The computed learning rate decay function.
        """
        global_step = tf.to_int32(global_step)

        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(
            learning_rate=learning_rate,
            global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name="decayed_learning_rate")

        final_lr = tf.train.piecewise_constant(
            x=global_step,
            boundaries=[start_decay_at],
            values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)

        return final_lr

    return decay_fn


def create_input_fn(pipeline,
                    batch_size,
                    bucket_boundaries=None,
                    allow_smaller_final_batch=False,
                    mode=None,
                    scope=None):
    """Creates an input function that can be used with tf.learn estimators.
      Note that you must pass "factory funcitons" for both the data provider and
      featurizer to ensure that everything will be created in  the same graph.

    Args:
      pipeline: An instance of `seq2seq.data.InputPipeline`.
      batch_size: Create batches of this size. A queue to hold a
        reasonable number of batches in memory is created.
      bucket_boundaries: int list, increasing non-negative numbers.
        If None, no bucket is performed.

    Returns:
      An input function that returns `(feature_batch, labels_batch)`
      tuples when called.
    """

    def input_fn():
        """Creates features and labels.
        """

        with tf.variable_scope(scope or "input_fn"):
            data_provider = pipeline.make_data_provider()
            features_and_labels = pipeline.read_from_data_provider(data_provider)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                is_training = True
                trainable = True
            elif mode == tf.contrib.learn.ModeKeys.EVAL:
                is_training = False
                trainable = False
            else:
                is_training = False
                trainable = False

            thread_id = random.randint(0, 3)
            features_and_labels["source_tokens"] = process_video(features_and_labels["source_tokens"],
                                                                 features_and_labels["source_len"],
                                                                 is_training=is_training,
                                                                 height=240,
                                                                 early_fusion_value=5,
                                                                 thread_id=thread_id,
                                                                 frame_format="jpeg")
            features_and_labels["source_tokens"] = vgg_m(
                features_and_labels["source_tokens"],
                trainable=trainable,
                is_training=is_training)

            features_and_labels.pop("record_key")
            if bucket_boundaries:
                _, batch = tf.contrib.training.bucket_by_sequence_length(
                    input_length=features_and_labels["source_len"],
                    bucket_boundaries=bucket_boundaries,
                    tensors=features_and_labels,
                    batch_size=batch_size,
                    keep_input=features_and_labels["source_len"] >= 1,
                    dynamic_pad=True,
                    capacity=5000 + 16 * batch_size,
                    allow_smaller_final_batch=allow_smaller_final_batch,
                    name="bucket_queue")
            else:
                batch = tf.train.batch(
                    tensors=features_and_labels,
                    enqueue_many=False,
                    batch_size=batch_size,
                    dynamic_pad=True,
                    capacity=5000 + 16 * batch_size,
                    allow_smaller_final_batch=allow_smaller_final_batch,
                    name="batch_queue")

            # Separate features and labels
            features_batch = {k: batch[k] for k in pipeline.feature_keys}
            if set(batch.keys()).intersection(pipeline.label_keys):
                labels_batch = {k: batch[k] for k in pipeline.label_keys}
            else:
                labels_batch = None

            return features_batch, labels_batch

    return input_fn


def process_video(encoded_video,
                  video_length,
                  is_training,
                  height,
                  early_fusion_value,
                  resize_height=120,
                  resize_width=120,
                  thread_id=0,
                  frame_format="jpeg"):
    """Decode a video, resize, apply random distortions and do early fusion.

    In training, frames are distorted slightly differently depending on thread_id.

    Args:
      encoded_video: String Array Tensor containing the video frames.
      is_training: Boolean; whether preprocessing for training or eval.
      height: Height of the output image.
      width: Width of the output image.
      early_fusion_value: number of frames to fuse
      resize_height: If > 0, resize height before crop to final dimensions.
      resize_width: If > 0, resize width before crop to final dimensions.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions. There should be a multiple of 2 preprocessing threads.
      frame_format: "jpeg" or "png".

    Returns:
      A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

    Raises:
      ValueError: If image_format is invalid.
    """

    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    # def image_summary(name, image):
    #     if not thread_id:
    #         tf.summary.image(name, tf.expand_dims(image, 0))

    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
    with tf.name_scope("decode", values=[encoded_video]):
        input_jpeg_strings = tf.TensorArray(tf.string, video_length)
        input_jpeg_strings = input_jpeg_strings.unstack(encoded_video)
        init_array = tf.TensorArray(tf.float32, size=video_length)

        def cond(i, ta):
            return tf.less(i, video_length)

        def body(i, ta):
            image = input_jpeg_strings.read(i)
            image = tf.image.decode_jpeg(image, 3, name='decode_image')
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            assert (resize_height > 0) == (resize_width > 0)
            image = tf.image.resize_images(image, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)
            return i + 1, ta.write(i, image)

        _, input_image = tf.while_loop(cond, body, [0, init_array])

        video = input_image.stack()
        num_frames = tf.shape(video)[0]
        num_of_fusions = tf.div(num_frames, early_fusion_value)
        end_frame_index = num_of_fusions * early_fusion_value
        video = video[0:end_frame_index, :, :, :]
        if is_training:
            video = distort_video(video, thread_id)

        # Rescale to [-1,1] instead of [0, 1]
        video = tf.map_fn(lambda x: tf.subtract(x, 0.5), video)
        video = tf.map_fn(lambda x: tf.multiply(x, 2.0), video)

        # convert to grayscale
        video = tf.map_fn(lambda x: tf.image.rgb_to_grayscale(x), video)

        # reshape video for early fusion
        video = tf.reshape(video, [num_of_fusions, early_fusion_value, resize_height, resize_width, 1])

        # fuse the frames
        video = tf.map_fn(lambda x: tf.concat(tf.unstack(x), 2), video)

        return video


def distort_video(video, thread_id):
    """Perform random distortions on a video.

    Args:
      video: A float32 Tensor of shape [frame_num, height, width, 1] with values in [0, 1).
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions. There should be a multiple of 2 preprocessing threads.

    Returns:
      distorted_image: A float32 Tensor of shape [height, width, 3] with values in
        [0, 1].
    """

    # Get random factors once to apply to all video frames
    brightness_delta = tf.random_uniform([], -32. / 255., 32. / 255.)
    contrast_factor = tf.random_uniform([], 0.5, 1.5)
    hue_delta = tf.random_uniform([], -0.032, 0.032)
    saturation_factor = tf.random_uniform([], 0.5, 1.5)
    flip_factor = tf.random_uniform([], 0, 1.0)

    # Randomly flip horizontally.
    with tf.name_scope("flip_horizontal", values=[video]):
        video = tf.map_fn(lambda x: random_flip_left_right(x, flip_factor=flip_factor), video)

    # Randomly distort the colors based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[video]):
        if color_ordering == 0:
            video = tf.map_fn(lambda x: random_brightness(x, delta=brightness_delta), video)
            video = tf.map_fn(lambda x: random_saturation(x, saturation_factor=saturation_factor), video)
            video = tf.map_fn(lambda x: random_hue(x, delta=hue_delta), video)
            video = tf.map_fn(lambda x: random_contrast(x, contrast_factor=contrast_factor), video)
        elif color_ordering == 1:
            video = tf.map_fn(lambda x: random_brightness(x, delta=brightness_delta), video)
            video = tf.map_fn(lambda x: random_contrast(x, contrast_factor=contrast_factor), video)
            video = tf.map_fn(lambda x: random_saturation(x, saturation_factor=saturation_factor), video)
            video = tf.map_fn(lambda x: random_hue(x, delta=hue_delta), video)

        # The random_* ops do not necessarily clamp.
        video = tf.map_fn(lambda x: tf.clip_by_value(x, 0.0, 1.0), video)

    return video


def random_brightness(image, delta):
    with tf.name_scope(None, 'adjust_brightness', [image, delta]) as name:
        image = tf.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, tf.float32)
        adjusted = tf.add(flt_image, tf.cast(delta, tf.float32), name=name)
        adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
        return tf.image.convert_image_dtype(adjusted, orig_dtype, saturate=True)


def random_contrast(image, contrast_factor):
    with tf.name_scope(None, 'adjust_contrast',
                       [image, contrast_factor]) as name:
        images = tf.convert_to_tensor(image, name='images')
        # Remember original dtype to so we can convert back if needed
        orig_dtype = images.dtype
        flt_images = tf.image.convert_image_dtype(images, tf.float32)
        # pylint: disable=protected-access
        adjusted = _adjust_contrastv2(flt_images, contrast_factor=contrast_factor, name=name)
        # pylint: enable=protected-access
        return tf.image.convert_image_dtype(adjusted, orig_dtype, saturate=True)


def random_hue(image, delta):
    with tf.name_scope(None, 'adjust_hue', [image]) as name:
        image = tf.convert_to_tensor(image, name='image')
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, tf.float32)

        # TODO(zhengxq): we will switch to the fused version after we add a GPU
        # kernel for that.
        fused = os.environ.get('TF_ADJUST_HUE_FUSED', '')
        fused = fused.lower() in ('true', 't', '1')

        if not fused:
            hsv = tf.image.rgb_to_hsv(flt_image)

            hue = tf.slice(hsv, [0, 0, 0], [-1, -1, 1])
            saturation = tf.slice(hsv, [0, 0, 1], [-1, -1, 1])
            value = tf.slice(hsv, [0, 0, 2], [-1, -1, 1])

            # Note that we add 2*pi to guarantee that the resulting hue is a positive
            # floating point number since delta is [-0.5, 0.5].
            hue = tf.mod(hue + (delta + 1.), 1.)

            hsv_altered = tf.concat([hue, saturation, value], 2)
            rgb_altered = tf.image.hsv_to_rgb(hsv_altered)
        else:
            rgb_altered = tf.image.adjust_hue(flt_image, delta)

        return tf.image.convert_image_dtype(rgb_altered, orig_dtype)


def random_saturation(image, saturation_factor):
    with tf.name_scope(None, 'adjust_saturation', [image]) as name:
        image = tf.convert_to_tensor(image, name='image')
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, tf.float32)

        # TODO(zhengxq): we will switch to the fused version after we add a GPU
        # kernel for that.
        fused = os.environ.get('TF_ADJUST_SATURATION_FUSED', '')
        fused = fused.lower() in ('true', 't', '1')

        if fused:
            return tf.image.convert_image_dtype(
                tf.image.adjust_saturation(flt_image, saturation_factor),
                orig_dtype)

        hsv = tf.image.rgb_to_hsv(flt_image)

        hue = tf.slice(hsv, [0, 0, 0], [-1, -1, 1])
        saturation = tf.slice(hsv, [0, 0, 1], [-1, -1, 1])
        value = tf.slice(hsv, [0, 0, 2], [-1, -1, 1])

        saturation *= saturation_factor
        saturation = tf.clip_by_value(saturation, 0.0, 1.0)

        hsv_altered = tf.concat([hue, saturation, value], 2)
        rgb_altered = tf.image.hsv_to_rgb(hsv_altered)

        return tf.image.convert_image_dtype(rgb_altered, orig_dtype)


def random_flip_left_right(image, flip_factor):
    image = tf.convert_to_tensor(image, name='image')
    mirror_cond = tf.less(flip_factor, .5)
    result = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    return result


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_m_base(inputs,
               scope=None):
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.conv2d(inputs, 96, [3, 3], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv3')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 512, [6, 6], padding='VALID', scope='fc6')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


def vgg_m(video,
          trainable=True,
          is_training=True,
          weight_decay=0.00004,
          stddev=0.1,
          dropout_keep_prob=0.8,
          use_batch_norm=True,
          batch_norm_params=None,
          add_summaries=True,
          scope="VGG-M"):
    # Only consider the inception model to be in training mode if it's trainable.
    is_vgg_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_vgg_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    with tf.variable_scope(scope, "VGG-M", [video]) as scope:
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=trainable):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = vgg_m_base(video, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = slim.dropout(
                        net,
                        keep_prob=dropout_keep_prob,
                        is_training=is_vgg_model_training,
                        scope="dropout")
                    net = slim.flatten(net, scope="flatten")

    # Add summaries.
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net
