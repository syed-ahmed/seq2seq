from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading

from nltk.tokenize.moses import MosesTokenizer
import fnmatch
import numpy as np
import pandas as pd
import tensorflow as tf

tf.flags.DEFINE_string("train_video_dir", "/Users/luna/workspace/ASLNet/data/raw-data/frames",
                       "Training video directory.")

tf.flags.DEFINE_string("dataset_info_file", "/Users/luna/workspace/ASLNet/data/raw-data/info.csv",
                       "Dataset metadata")

tf.flags.DEFINE_string("output_dir", "/Users/luna/workspace/ASLNet/data/processed-data", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")

tf.flags.DEFINE_float("train_cutoff", 0.95,
                      "Training dataset percentage split.")

tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")

tf.flags.DEFINE_integer("min_word_count", 1,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/Users/luna/workspace/ASLNet/data/processed-data/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

VideoMetadata = namedtuple("VideoMetadata",
                           ["video_id", "video_type", "filename", "captions", "width", "height", "frame_count",
                            "fps", "duration"])


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


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

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    if type(value) is unicode:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value.encode('utf-8'))]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(video, decoder):
    """Builds a SequenceExample proto for an video-caption pair.

    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.

    Returns:
      A SequenceExample proto.
    """
    frames = sorted(_find_files(video.filename, "*.jpg"), key=lambda x: int(filter(str.isdigit, x.split("/")[-1])))
    caption = video.captions
    feature = {
        "video/video_id": _int64_feature(video.video_id),
        "video/video_type": _bytes_feature(video.video_type),
        "video/filename": _bytes_feature(video.filename.split("/")[-1]),
        "video/width": _int64_feature(video.width),
        "video/height": _int64_feature(video.height),
        "video/frame_count": _int64_feature(video.frame_count),
        "video/fps": _int64_feature(video.fps),
        "video/duration": _float_feature(video.duration),
    }
    context = tf.train.Features(feature=feature)
    frames_encoded = []
    for idx, val in enumerate(frames):
        with tf.gfile.FastGFile(val, "r") as f:
            encoded_image = f.read()
        try:
            decoder.decode_jpeg(encoded_image)
            frames_encoded.append(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % val)
            return

    feature_lists = tf.train.FeatureLists(feature_list={
        "video/caption": _bytes_feature_list(caption),
        "video/frames": _bytes_feature_list(frames_encoded)

    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_video_files(thread_index, ranges, name, videos, decoder,
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

            sequence_example = _to_sequence_example(video, decoder)
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
    # Get videos of word type and sentence type (refer to lip net paper - part of curriculum
    # training. makes the model converge faster).
    word_videos = [video for video in videos if video.video_type == "WORD"]
    sentence_videos = [video for video in videos if video.video_type == "SENTENCE"]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(word_videos)
    random.seed(12345)
    random.shuffle(sentence_videos)

    word_videos.extend(sentence_videos)

    # Break the videos into num_threads batches. Batch i is defined as
    # word_videos[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(word_videos), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, word_videos, decoder, num_shards)
        t = threading.Thread(target=_process_video_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d video-caption pairs in data set '%s'." %
          (datetime.now(), len(word_videos), name))


def _create_vocab(captions):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
      captions: A list of lists of strings.

    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s\t%d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)


def _process_caption(caption):
    """Processes a caption string into a list of tokenized words.

    Args:
      caption: A string caption.

    Returns:
      A list of strings; the tokenized caption.
    """
    tokenizer = MosesTokenizer()
    tokenized_caption = ["SEQUENCE_START"]
    tokenized_caption.extend(tokenizer.tokenize(caption.lower()))
    tokenized_caption.append("SEQUENCE_END")
    return tokenized_caption


def _load_and_process_metadata(dataset_metadata, video_dir):
    """Loads video metadata from a csv file and processes the captions.

    Args:
      dataset_metadata: CSV file containing video metadata
      video_dir: Directory containing the video frames.

    Returns:
      A list of ImageMetadata.
    """
    video_metadata = pd.read_csv(dataset_metadata)

    # Extract data.
    pd_video_id = video_metadata["ID"].tolist()
    pd_video_type = video_metadata["Video Type"].tolist()
    pd_filename = video_metadata["names"].tolist()
    pd_captions = video_metadata["Captions"].tolist()
    pd_width = video_metadata["Video Width"].tolist()
    pd_height = video_metadata["Video Height"].tolist()
    pd_frame_count = video_metadata["Frame Count"].tolist()
    pd_fps = video_metadata["FPS"].tolist()
    pd_duration = video_metadata["duration"].tolist()

    assert len(pd_video_id) == len(pd_video_type) \
           == len(pd_filename) == len(pd_captions) == len(pd_width) == len(pd_height) \
           == len(pd_frame_count) == len(pd_fps) == len(pd_duration)

    print("Loaded all the metadata...")

    # Process the captions and combine the data into a list of VideoMetadata.
    print("Proccessing captions.")
    video_metadata = []
    num_captions = 0
    for idx, val in enumerate(pd_filename):
        filename = os.path.join(video_dir, val)
        captions = _process_caption(pd_captions[idx].decode('utf-8'))
        video_id = pd_video_id[idx]
        video_type = pd_video_type[idx]
        width = int(pd_width[idx])
        height = int(pd_height[idx])
        fps = int(pd_fps[idx])
        frame_count = int(pd_frame_count[idx])
        duration = pd_duration[idx]

        video_metadata.append(VideoMetadata(video_id, video_type, filename, captions, width, height, frame_count,
                                            fps, duration))
        num_captions += len(captions)
    print("Finished processing %d captions for %d videos in %s" %
          (num_captions, len(pd_filename), dataset_metadata))

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
    dataset = _load_and_process_metadata(FLAGS.dataset_info_file,
                                         FLAGS.train_video_dir)

    # Redistribute the data as follows:
    #   train_dataset = 95%
    #   val_dataset = 5%
    # leaving only sentences for validation
    val_cutoff = int((1-FLAGS.train_cutoff) * len(dataset))
    val_dataset = dataset[0:val_cutoff]
    train_dataset = dataset[val_cutoff:]

    # Create vocabulary from the training captions.

    train_captions = [video.captions for video in train_dataset]
    _create_vocab(train_captions)

    _process_dataset("train", train_dataset, FLAGS.train_shards)
    _process_dataset("val", val_dataset, FLAGS.val_shards)


if __name__ == "__main__":
    tf.app.run()
