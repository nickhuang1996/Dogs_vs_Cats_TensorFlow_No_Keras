import tensorflow as tf


def get_decay_lr(args, train_samples, global_step):
    """

    Every 10 epochs decay learning rate.
    """
    num_batches_per_epoch = train_samples / args.batch_size
    # decay_steps = int(num_batches_per_epoch * 10)
    decay_steps = num_batches_per_epoch

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        learning_rate=args.lr,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=args.decay_rate,
        staircase=True,
    )
    return lr
