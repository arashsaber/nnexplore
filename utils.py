#!/usr/bin/python3
"""
The file contains some utility functions for tensorflow.

Copyright (c) 2017
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

import matplotlib.pyplot as plt
import os
import tensorflow as tf



class tfRecordBuilder(object):

    def __init__(self,
                 dataset_path, annotation_path,
                 tfRecord_path, file_format='jpg'):

        self.dataset_path = dataset_path
        self.annotation_path = annotation_path
        self.tfRecord_path = tfRecord_path
        self.file_format = file_format.split()
        self._coder = ImageCoder()
        self._KAIST_converter()

    def _KAIST_image(self, thermal_image_path, rgb_image_path, label_path):
        """
        Given the addresses of the visible, thermal and label data, the function extracts the
        images and the labels
        :param thermal_image_path: string, address of the thermal image
        :param rgb_image_path: string, address of the visible image
        :param label_path: string, address of the annotation data
        :return:
        """

        # Read the image file.
        thermal_image_data = tf.gfile.FastGFile(thermal_image_path, 'r').read() #maybe 'rb'
        rgb_image_data = tf.gfile.FastGFile(rgb_image_path, 'r').read()

        image = self._coder.decode_jpeg(rgb_image_data) # should we use tf.get_shape()

        # Read and parse the label
        labels = []
        boxes = []
        with open(label_path) as label_file:
            label_text = label_file.read()
            parsed_text = label_text.split(' 0 0 0 0 0 0 0')
            parsed_text[0] = parsed_text[0].split('version=3')
            for i in range(len(parsed_text)):
                object_label = parsed_text[i].split()
                labels.append(object_label[0])
                boxes.append([float(val) for val in object_label[1:]])

        bboxes = []
        for i in range(len(boxes)):
            bboxes.append((float(boxes[i][0].find('ymin').text) / image.shape[0],
                           float(boxes[i][0].find('xmin').text) / image.shape[1],
                           float(boxes[i][0].find('ymax').text) / image.shape[0],
                           float(boxes[i][0].find('xmax').text) / image.shape[1]
                           ))

        return rgb_image_data, thermal_image_data, bboxes, labels, image.shape

    def _convert_to_example(self,
                            rgb_image_data, thermal_image_data,
                            bboxes, labels, shape):
        """
        convert the data into tensorflow example structure
        :param rgb_image_data:
        :param thermal_image_data:
        :param bboxes:
        :param labels:
        :param shape:
        :return:
        """
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for b in bboxes:
            assert len(b) == 4
            [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self._int64_feature(shape[0]),
            'image/width': self._int64_feature(shape[1]),
            'image/channels': self._int64_feature(shape[2]),
            'image/shape': self._int64_feature(shape),
            'image/object/bbox/xmin': self._float_feature(xmin),
            'image/object/bbogix/xmax': self._float_feature(xmax),
            'image/object/bbox/ymin': self._float_feature(ymin),
            'image/object/bbox/ymax': self._float_feature(ymax),
            'image/object/bbox/label': self._int64_feature(labels),
            'image/thermal': self._bytes_feature(thermal_image_data),
            'image/rgb': self._bytes_feature(rgb_image_data)}))

        return example

    def _KAIST_converter(self):

        if not tf.gfile.Exists(self.dataset_path):
            raise ValueError('dataset_path does not exist')
        if not tf.gfile.Exists(self.tfRecord_path):
            tf.gfile.MakeDirs(self.tfRecord_path)

        fidx = 0
        for (dirpath, dirnames, filenames) in os.walk(self.dataset_path):

            print(dirpath, dirnames, filenames)
            path = os.path.split(dirpath)
            label_dir = os.path.join(self.parent_directory(path, 1),
                                     self.parent_directory(path, 0)) #os.path.split(path[0])[-1]

            if filenames and path[-1] == 'lwir':
                output_dir = os.path.join(self.tfRecord_path, self.parent_directory(path, 1))

                tf_filename = self._get_output_filename(output_dir, self.parent_directory(path, 0), fidx)
                fidx += 1
                with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:

                    for image_file in filenames:
                        if image_file.endswith(self.file_format):
                            thermal_image_path = os.path.join(dirpath, image_file)
                            rgb_image_path = os.path.join(path[0], 'visible', image_file)
                            label_path = os.path.join(self.annotation_path, label_dir,
                                                      image_file.replace(self.file_format, 'txt'))
                            rgb_image_data, thermal_image_data, bboxes, labels, shape = \
                                self._KAIST_image(thermal_image_path, rgb_image_path, label_path)
                            example = self._convert_to_example(rgb_image_data, thermal_image_data,
                                                               bboxes, labels, shape)
                            tfrecord_writer.write(example.SerializeToString())


    @staticmethod
    def parent_directory(path, n):
        while n > 0:
            parsed = os.path.split(path)
            path = parsed[0]
            n -= 1
        parsed = os.path.split(path)
        return parsed[-1]

    @staticmethod
    def _get_output_filename(output_dir, name, idx):
        return '{:s}/{:s}_{:05d}.tfrecord'.format(output_dir, name, idx)

    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting int64 features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _float_feature(value):
        """Wrapper for inserting float features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


#   ------------------------------------
#Create an image reader object for easy reading of the images
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

#   ------------------------------------
if __name__ == '__main__':
    dataset_path = ''
    annotation_path = ''
    tfRecord_path = ''
    Z = tfRecordBuilder(dataset_path, annotation_path, tfRecord_path)



