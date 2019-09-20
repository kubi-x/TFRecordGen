"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import hashlib

flags = tf.app.flags
flags.DEFINE_string('csv_input_train', 'csv/train_labels.csv', 'Path to the CSV train input')
flags.DEFINE_string('output_path_train', 'tfrecords/train.record', 'Path to output train TFRecord')
flags.DEFINE_string('image_dir_train', 'data/train/images/', 'Path to train images')

flags.DEFINE_string('csv_input_validation', 'csv/validation_labels.csv', 'Path to the CSV validation input')
flags.DEFINE_string('output_path_validation', 'tfrecords/validation.record', 'Path to output validation TFRecord')
flags.DEFINE_string('image_dir_validation', 'data/validation/images/', 'Path to validation images')

flags.DEFINE_string('csv_input_test', 'csv/labels.csv', 'Path to the CSV test input')
flags.DEFINE_string('output_path_test', 'tfrecords/test.record', 'Path to output test TFRecord')
flags.DEFINE_string('image_dir_test', 'data/test/images/', 'Path to test images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'phone':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    key = hashlib.sha256(encoded_jpg).hexdigest()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate(csv_input, image_dir, output_path):
    try:
        writer = tf.python_io.TFRecordWriter(output_path)
        path = os.path.join(image_dir)
        examples = pd.read_csv(csv_input)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))
    except Exception as err:
        os.remove(output_path)
        raise err


def main(_):
    # generate(FLAGS.csv_input_train, FLAGS.image_dir_train, FLAGS.output_path_train)
    # generate(FLAGS.csv_input_validation, FLAGS.image_dir_validation, FLAGS.output_path_validation)
    # generate(FLAGS.csv_input_test, FLAGS.image_dir_test, FLAGS.output_path_test)
    pass


if __name__ == '__main__':
    tf.app.run()
