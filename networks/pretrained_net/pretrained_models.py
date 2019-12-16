from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim
from networks.pretrained_net import vgg_preprocessing
import tensorflow as tf


def preprocess(args, image, is_training=True):
    if args.pretrained_model == 'resnet_v1_50':
        processed_image = vgg_preprocessing.preprocess_image(
            image=image,
            output_height=args.image_size[0],
            output_width=args.image_size[1],
            is_training=is_training,
        )
    else:
        raise NotImplementedError('Unsupported Pretrained Model {}'.format(args.pretrained_model))
    return processed_image


def parse_and_preprocess_data(args, example_proto, is_training):
    features = {
        'img_raw': tf.FixedLenFeature([], tf.string, ''),
        'label': tf.FixedLenFeature([], tf.int64, 0)
    }
    parsed_features = tf.parse_single_example(
        serialized=example_proto,
        features=features,
    )
    image = tf.image.decode_jpeg(parsed_features['img_raw'], channels=3)
    label = tf.cast(parsed_features['label'], dtype=tf.int64)
    image = tf.cast(image, dtype=tf.float32)
    processed_image = preprocess(args=args, image=image, is_training=is_training)
    return processed_image, label


# model
def inference(args, processed_images, class_num, is_training):
    if args.pretrained_model == 'resnet_v1_50':
        print("load model: resnet_v1_50")
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, endpoints = resnet_v1.resnet_v1_50(
                processed_images,
                num_classes=None, #class_num,
                is_training=is_training,
            )
        net = tf.squeeze(net, [1, 2])
        logits = slim.fully_connected(net,
                                      num_outputs=class_num,
                                      activation_fn=None)
    else:
        raise NotImplementedError('Unsupported Pretrained Model {}'.format(args.pretrained_model))
    return logits
