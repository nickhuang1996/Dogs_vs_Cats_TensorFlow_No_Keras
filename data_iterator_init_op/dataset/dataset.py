import tensorflow as tf
from data_iterator_init_op.dataset.DC import DC
from networks.pretrained_net.pretrained_models import parse_and_preprocess_data


def get_dataset(args):
    dataset = DC(args=args)
    training_dataset = tf.data.TFRecordDataset(dataset.train_record_dir)
    training_dataset = training_dataset.map(
        lambda example: parse_and_preprocess_data(
            args=args,
            example_proto=example,
            is_training=True,
        )
    )
    training_dataset = training_dataset.batch(args.batch_size).repeat()

    validation_dataset = tf.data.TFRecordDataset(dataset.validation_record_dir)
    validation_dataset = validation_dataset.map(
        lambda example: parse_and_preprocess_data(
            args=args,
            example_proto=example,
            is_training=False,
        )
    )
    validation_dataset = validation_dataset.batch(args.batch_size)
    return dataset, training_dataset, validation_dataset


