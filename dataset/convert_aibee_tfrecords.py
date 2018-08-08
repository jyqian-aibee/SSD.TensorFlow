from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random
import sys
import threading
import numpy as np
import six
import tensorflow as tf
import argparse
import json


# =========================Feature function============================
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# =======================================================================


def load_dataset(dataset_path, num_threads):
    """
    Load coco format dataset, and split into several small dictionaries for each thread to use.
    """
    big_dict = dict()
    output_dicts = []
    with open(dataset_path, 'r', encoding='latin-1') as f:
        dataset = json.load(f)
    for instance in dataset['annotations']:
        img_id = instance['image_id']
        if dataset['images'][img_id]['file_name'] not in big_dict:
            big_dict[dataset['images'][img_id]['file_name']] = []
        big_dict[dataset['images'][img_id]['file_name']].append(instance['bbox'])
    len_instances = len(big_dict)
    pics_per_dict = len_instances // num_threads
    for i in range(num_threads):
        if i != num_threads - 1:
            dic = dict(list(big_dict.items())[i * pics_per_dict:(i + 1) * pics_per_dict])
        else:
            dic = dict(list(big_dict.items())[i * pics_per_dict:])
        output_dicts.append(dic)
    lens = [len(dic) for dic in output_dicts]
    print('Generated {} dicts, with size {}'.format(len(output_dicts), lens))
    return output_dicts


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def create_tf_example(filename, image_buffer, bboxes, labels, height, width):
    height = height  # Image height
    width = width  # Image width
    filename = filename  # Filename of the image. Empty if image is not from file
    encoded_image_data = image_buffer  # Encoded image bytes
    image_format = 'JPEG'  # b'jpeg' or b'png'
    channels = 3

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box

    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([xmins, ymins, xmaxs, ymaxs], b)]
    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/filename': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/format': _bytes_feature(image_format),
        'image/object/bbox/xmin': _float_feature(xmins),
        'image/object/bbox/xmax': _float_feature(xmaxs),
        'image/object/bbox/ymin': _float_feature(ymins),
        'image/object/bbox/ymax': _float_feature(ymaxs),
        'image/object/class/label': _int64_feature(labels),
    }))
    return tf_label_and_data


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _thread_func(coder, thread_meta, subset, directory, num_shards, name='Train'):
    num_pictures = len(subset.keys())
    thread_index, total_threads = thread_meta
    pictures_per_shard = int(num_pictures // num_shards)
    items = list(subset.items())
    for shard_index in range(num_shards):
        start = time.time()
        shard = thread_index * num_shards + shard_index
        output_name = '{}-{:03d}-of-{:03d}'.format(name, shard, total_threads * num_shards)
        output_file = os.path.join(directory, output_name)
        writer = tf.python_io.TFRecordWriter(output_file)
        if shard_index != num_shards - 1:
            items_to_shard = items[shard_index * pictures_per_shard: (shard_index + 1) * pictures_per_shard]
        else:
            items_to_shard = items[shard_index * pictures_per_shard:]
        for i, item in enumerate(items_to_shard):
            filename, bboxes = item
            labels = [1] * len(bboxes)
            image_buffer, height, width = _process_image(filename, coder)
            example = create_tf_example(filename, image_buffer, bboxes, labels, height, width)
            writer.write(example.SerializeToString())

            if i % 1000 == 0:
                print('Thread {}: processed {} images.'.format(thread_index, i))
        stop = time.time()
        writer.close()
        print('Thread {}: wrote {} images to {}. Time: {:.2f}'.format(thread_index, len(items_to_shard), output_file,
                                                                      stop - start))


def convert_dataset_to_tfrecords(dataset_dir, out_dir, num_threads=8, name='Train'):
    # Load dataset
    dataset_list = load_dataset(dataset_dir, num_threads)
    coord = tf.train.Coordinator()
    coder = ImageCoder()

    threads = []
    for thread_index in range(num_threads):
        args = (coder, [thread_index, num_threads], dataset_list[thread_index], out_dir, 4, name)
        t = threading.Thread(target=_thread_func, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('Finished writing all images into tfrecords.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("coco", type=str, help="coco file to be transferred to tfrecords")
    ap.add_argument("output", type=str, help="output root directory")
    ap.add_argument("--name", type=str, default="aibee-train", required=False, help="output tfrecords name prefix")
    ap.add_argument("-t", "--threads", type=int, default=4, required=False, help="number of threads to work")
    args = ap.parse_args()
    convert_dataset_to_tfrecords(args.coco, args.output, args.threads)
