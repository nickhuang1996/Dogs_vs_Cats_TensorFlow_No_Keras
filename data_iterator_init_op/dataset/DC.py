import os
import os.path as osp
import numpy as np
import tensorflow as tf
import sys


class DC(object):
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.record_dir = args.record_dir
        self.image_label_list_file = args.image_label_list_file
        self.train_data_ratio = args.train_data_ratio
        self.train_record_dir = osp.join(self.record_dir, 'train_img.tfrecord')
        self.validation_record_dir = osp.join(self.record_dir, 'validation_img.tfrecord')

        self.image_list = []
        self.label_list = []
        self.train_images = 0
        self.validation_images = 0
        self.get_list()
        self.create_image_label_list()
        self.image2tf_record()
        self.item = {
            'image_list': self.image_list,
            'label_list': self.label_list,
        }

    def get_list(self):
        for filename in os.listdir(self.data_dir):
            name = filename.split('.')
            self.image_list.append(osp.join(self.data_dir, filename))
            if name[0] == 'cat':
                self.label_list.append(0)
            else:
                self.label_list.append(1)
        image_label_list = self.shuffle()
        self.image_list = list(image_label_list[:, 0])
        self.label_list = list(image_label_list[:, 1])
        self.label_list = [int(i) for i in self.label_list]

        self.train_images = self.get_train_images()
        self.validation_images = self.get_validation_images()

    def get_train_images(self):
        return int(self.train_data_ratio * len(self.image_list))

    def get_validation_images(self):
        return len(self.image_list) - int(self.train_data_ratio * len(self.image_list))

    def shuffle(self):
        image_label_list = np.array([self.image_list,
                                     self.label_list])
        image_label_list = image_label_list.transpose()
        np.random.shuffle(image_label_list)
        return image_label_list

    def create_image_label_list(self):
        with open(self.image_label_list_file, 'w') as f:
            for i in range(len(self.image_list)):
                f.write(self.image_list[i] + '\t\t' + str(self.label_list[i]) + '\n')

    def image2tf_record(self):
        train_tf_record_writer = tf.python_io.TFRecordWriter(path=self.train_record_dir)
        validation_tf_record_writer = tf.python_io.TFRecordWriter(path=self.validation_record_dir)
        print("Start TFRecord writing...")
        self.image_list[:self.train_images], \
            self.label_list[:self.train_images] = self.tf_record_write(
            tf_record_writer=train_tf_record_writer,
            image_list=self.image_list[:self.train_images],
            label_list=self.label_list[:self.train_images])
        self.image_list[self.train_images:], \
            self.label_list[self.train_images:] = self.tf_record_write(
            tf_record_writer=validation_tf_record_writer,
            image_list=self.image_list[self.train_images:],
            label_list=self.label_list[self.train_images:])
        train_tf_record_writer.close()
        validation_tf_record_writer.close()
        print("TFRecord writing is over.")

    @staticmethod
    def tf_record_write(tf_record_writer, image_list, label_list):
        print("length=", len(image_list))
        num = 0
        for image, label in zip(image_list, label_list):
            with open(image, 'rb') as f:
                encoded_jpg = f.read()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])
                        ),
                        'img_raw': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[encoded_jpg])
                        )
                    }
                )
            )
            tf_record_writer.write(example.SerializeToString())
            num += 1
            sys.stdout.write("\rIndex {} has been written.".format(num))
            sys.stdout.flush()
        print("\n")
        return image_list, label_list


if __name__ == '__main__':
    from parser_setting import args
    dataset = DC(args=args)
