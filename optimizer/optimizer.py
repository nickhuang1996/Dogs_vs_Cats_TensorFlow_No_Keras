import tensorflow as tf
from optimizer.decay_lr import get_decay_lr


def get_optimizer(args, train_samples, global_step):
    lr = get_decay_lr(args=args,
                      train_samples=train_samples,
                      global_step=global_step)

    if args.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
        )
    elif args.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr,
        )
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(args.optimizer))
    return optimizer
