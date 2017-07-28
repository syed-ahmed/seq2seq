from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from datetime import datetime
import os.path
import random
import sys
import threading
import fnmatch
import numpy as np
import tensorflow as tf
import ast

tf.flags.DEFINE_string("dataset_captions_file", "/Users/luna/workspace/seq2seq_v2/bin/data/train_with_id.tok",
                       "Dataset captions data")

tf.flags.DEFINE_string("dataset_features", "/Users/luna/workspace/seq2seq_v2/bin/data/",
                       "Dataset captions data")

tf.flags.DEFINE_string("output_dir", "/Users/luna/workspace/ASLNet/data/processed-data", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")

tf.flags.DEFINE_integer("num_curriculum", 10,
                        "Number of shards in training TFRecord files.")

tf.flags.DEFINE_float("train_cutoff", 0.95,
                      "Training dataset percentage split.")

tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

VideoMetadata = namedtuple("VideoMetadata",
                           ["filename", "features", "captions"])


def _find_files(directory, pattern):
    """
    Recursively finds all files matching the pattern.
    :param directory: directory of files
    :param pattern: file pattern to match
    :return: list of file paths
    """
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(values):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    if type(value) is unicode:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value.encode('utf-8'))]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _to_sequence_example(video):
    """Builds a SequenceExample proto for an video-caption pair.

    Args:
      video: An VideoMetadata object.
      decoder: An ImageDecoder object.

    Returns:
      A SequenceExample proto.
    """
    feature = {
        "video/filename": _bytes_feature(video.filename),
        "video/captions": _bytes_feature(video.captions),
    }
    context = tf.train.Features(feature=feature)

    feature_lists = tf.train.FeatureLists(feature_list={
        "video/features": _float_feature_list(video.features)

    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_video_files(thread_index, ranges, name, videos,
                         num_shards):
    """Processes and saves a subset of videos as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      videos: List of VideoMetadata.
      decoder: An ImageDecoder object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        videos_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in videos_in_shard:
            video = videos[i]

            sequence_example = _to_sequence_example(video)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d video-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d video-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, videos, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      num_shards: Integer number of shards for the output files.
    """

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(videos)

    # Break the videos into num_threads batches. Batch i is defined as
    # word_videos[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(videos), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, videos, num_shards)
        t = threading.Thread(target=_process_video_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d video-caption pairs in data set '%s'." %
          (datetime.now(), len(videos), name))


def _load_and_process_metadata(dataset_captions, dataset_features):
    """Loads video metadata from a csv file and processes the captions.

    Args:
      dataset_metadata: CSV file containing video metadata
      video_dir: Directory containing the video frames.

    Returns:
      A list of ImageMetadata.
    """
    captions_id_tuple = []

    with open(dataset_captions, "rb") as f:
        captions_with_id = f.readlines()

    for i in captions_with_id:
        captions_id_tuple.append((i.split("|")[0].strip(), i.split("|")[-1].strip()))
    # Process the captions and combine the data into a list of VideoMetadata.
    print("Proccessing captions.")
    video_metadata = []
    captions_id_tuple = [("video_10000", "hello")]
    for i in captions_id_tuple:
        key = i[0]
        feature_path = os.path.join(dataset_features, key + ".txt.vecs")
        with open(feature_path, "rb") as f:
            features = f.readlines()
            features = [x.strip("\n") for x in features]
            features = [x.split("\t") for x in features]

        feature_array = []
        for feature in features:
            array = ast.literal_eval(feature[1])
            array = [float(x) for x in array]
            feature_array.append(array)

        filename = features[0][0].split("/")[0]
        captions = i[1]
        video_metadata.append(VideoMetadata(filename, feature_array, captions))

    print("Finished processing %i video meta data in dataset" % len(video_metadata))

    return video_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    dataset = _load_and_process_metadata(FLAGS.dataset_captions_file,
                                         FLAGS.dataset_features)

    # Redistribute the data as follows:
    #   train_dataset = 95%
    #   val_dataset = 5%
    # leaving only sentences for validation
    val_cutoff = int((1 - FLAGS.train_cutoff) * len(dataset))
    val_dataset = random.sample(dataset, val_cutoff)
    train_dataset = [x for x in dataset if x not in val_dataset]

    # curriculum learning
    curr_spacing = np.linspace(0, len(train_dataset), FLAGS.num_curriculum + 1).astype(np.int)
    curr_ranges = []
    for i in xrange(len(curr_spacing) - 1):
        curr_ranges.append([curr_spacing[i], curr_spacing[i + 1]])

    for idx, val in enumerate(curr_ranges):
        _process_dataset("train-curriculum-" + str(idx), train_dataset[val[0]:val[1]], FLAGS.train_shards)

    # Create vocabulary from the training captions.

    _process_dataset("val", val_dataset, FLAGS.val_shards)


if __name__ == "__main__":
    tf.app.run()
