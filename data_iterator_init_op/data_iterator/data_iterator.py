import tensorflow as tf


def get_data_iterator(training_dataset):
    # training_dataset, validation_dataset = get_dataset(args=args)
    iterator = tf.data.Iterator.from_structure(
        output_types=training_dataset.output_types,
        output_shapes=training_dataset.output_shapes,
    )
    return iterator
