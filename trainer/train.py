from data_iterator_init_op.dataset.dataset import get_dataset
from data_iterator_init_op.data_iterator.data_iterator import get_data_iterator
from networks.pretrained_net.pretrained_models import inference
from loss import get_loss
from optimizer.get_apply_grad_op import get_apply_grad_op_group


import tensorflow as tf
from tensorflow.contrib import slim
import os.path as osp
import numpy as np
import math


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.log_dir = args.log_dir
        self.summary_writer = None

        self.dataset = None
        self.training_dataset = None
        self.validation_dataset = None
        self.training_init_op = None
        self.validation_init_op = None
        self.images = None
        self.labels = None

        self.variables_to_restore = None
        self.variables_to_train = None

        self.pretrained_models_dir = None
        self.pretrained_model = args.pretrained_model
        self.model_path = None
        self.checkpoint_path = args.checkpoint_dir + '/checkpoint/' + args.model_name
        self.pretrained_models_dir = args.checkpoint_dir + '/pretrained_models'

        self.per_evaluate_step = args.per_evaluate_step
        self.per_save_checkpoint_step = args.per_save_checkpoint_step
        self.max_step = args.max_step

        self.get_init_op()
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.logits = inference(args=args,
                                processed_images=self.images,
                                class_num=2,
                                is_training=self.is_training)
        self.pred = tf.nn.softmax(self.logits)
        self.top_k_op = tf.nn.in_top_k(self.logits, self.labels, 1)
        self.total_loss = get_loss(self.logits, self.labels)
        self.variables_to_restore_and_train()
        self.global_step = tf.train.get_or_create_global_step()

        self.train_samples = self.dataset.train_images
        self.validation_samples = self.dataset.validation_images

        self.apply_grad_operation_group = get_apply_grad_op_group(
            args=args,
            train_samples=self.train_samples,
            total_loss=self.total_loss,
            global_step=self.global_step,
            variables_to_train=self.variables_to_train,
            variables_to_restore=self.variables_to_restore,

        )

        self.get_model_path()

    def get_init_op(self):
        self.dataset, self.training_dataset, self.validation_dataset = get_dataset(args=self.args)
        iterator = get_data_iterator(training_dataset=self.training_dataset)
        self.training_init_op = iterator.make_initializer(self.training_dataset)
        self.validation_init_op = iterator.make_initializer(self.validation_dataset)
        self.images, self.labels = iterator.get_next()

    def variables_to_restore_and_train(self):
        if self.pretrained_model == 'resnet_v1_50':
            exclude = ['fully_connected']
            train_scope = ['fully_connected']

        else:
            exclude = []
            train_scope = []

        self.variables_to_restore = slim.get_variables_to_restore(
            exclude=exclude
        )
        self.variables_to_train = []
        for scope in train_scope:
            self.variables_to_train += slim.get_trainable_variables(scope=scope)

    def get_model_path(self):
        if self.pretrained_model == 'resnet_v1_50':
            self.model_path = self.pretrained_models_dir + '/' + self.pretrained_model + '.ckpt'
            assert osp.exists(self.model_path), "No such a directory or file {}.".format(self.model_path)
            print("train model path:", self.model_path)

    def train_initial(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

        init_fn = slim.assign_from_checkpoint_fn(
            model_path=self.model_path,
            var_list=self.variables_to_restore,
            ignore_missing_vars=True
        )
        self.summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        init_fn(session=sess)

        sess.run(self.training_init_op)
        print("Begin to train...")

    def save_initial_checkpoint(self, sess, saver):
        saver.save(sess, self.checkpoint_path, global_step=0)

    def train(self):
        with tf.Session() as sess:
            self.train_initial(sess=sess)
            saver = tf.train.Saver()
            self.save_initial_checkpoint(sess=sess, saver=saver)

            train_step = 0
            while train_step < self.max_step:
                _, train_loss, logits_op, pred_op, labels_op = sess.run(
                    [
                        self.apply_grad_operation_group,
                        self.total_loss,
                        self.logits,
                        self.pred,
                        self.labels,
                    ],
                    feed_dict={self.is_training: True}
                )
                train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)]
                )
                self.summary_writer.add_summary(train_summary, train_step)
                train_step += 1
                if train_step % self.per_evaluate_step == 0:
                    sess.run(self.validation_init_op)
                    precision = self.evaluate(sess=sess)
                    test_summary = tf.Summary(
                        value=[tf.Summary.Value(tag="precision", simple_value=precision)]
                    )

                    print("step: {}/{} loss: {:.4f}, validation precision: {}".format(train_step,
                                                                                      self.max_step,
                                                                                      train_loss,
                                                                                      precision))
                    self.summary_writer.add_summary(test_summary, train_step)
                    sess.run(self.training_init_op)
                if train_step % self.per_save_checkpoint_step == 0:
                    saver.save(sess, self.checkpoint_path, global_step=train_step)
                    print("step: {}/{} checkpoint has been saved.".format(train_step, self.max_step))
                if train_step == self.max_step and train_step % self.per_save_checkpoint_step != 0:
                    saver.save(sess, self.checkpoint_path, train_step)
                    print("step: {}/{}, loss: {}".format(train_step, self.max_step, train_loss))

    def evaluate(self, sess):
        iter_per_epoch = int(math.ceil(self.validation_samples / args.batch_size))

        correct_predict = 0
        step = 0

        while step < iter_per_epoch:
            predict = sess.run(self.top_k_op,
                               feed_dict={self.is_training: False})

            correct_predict += np.sum(predict)
            step += 1

        precision = correct_predict / self.validation_samples
        return precision





