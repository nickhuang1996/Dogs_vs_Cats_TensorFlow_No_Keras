from optimizer.optimizer import get_optimizer
import tensorflow as tf
from optimizer.decay_lr import get_decay_lr


def get_apply_grad_op(args, total_loss, train_samples, global_step):
    """

    temp_args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    """
    # Get optimizer
    optimizer = get_optimizer(args=args,
                              train_samples=train_samples,
                              global_step=global_step)
    # Compute gradients
    grads_and_vars = optimizer.compute_gradients(loss=total_loss)

    # Apply gradients
    apply_grad_operation = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
    )

    return apply_grad_operation


def get_apply_grad_op_group(args,
                            train_samples,
                            total_loss,
                            global_step,
                            variables_to_train,
                            variables_to_restore):
    lr = get_decay_lr(args=args,
                      train_samples=train_samples,
                      global_step=global_step)
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.01 * lr)

    grads = tf.gradients(total_loss, variables_to_train + variables_to_restore)
    grads1 = grads[:len(variables_to_train)]
    grads2 = grads[len(variables_to_train):]

    apply_grad_operation1 = optimizer1.apply_gradients(
        grads_and_vars=zip(grads1, variables_to_train),
        global_step=global_step
    )
    apply_grad_operation2 = optimizer2.apply_gradients(
        grads_and_vars=zip(grads2, variables_to_restore)
    )
    apply_grad_operation_group = tf.group(apply_grad_operation1, apply_grad_operation2)
    return apply_grad_operation_group
