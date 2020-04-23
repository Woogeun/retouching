"""
SPN train module
@authorized by Shasha Bae
@description: train the SPN model
"""

import random
import argparse
import os
from os.path import join
from glob import glob

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from utils import *
from network import *




def train_one_batch(inputs, labels, model, loss_fn, optimizer, loss, accuracy):
    with tf.GradientTape() as tape:
        # Calculate loss
        predictions = model(inputs)
        regularization_loss = tf.math.add_n(model.losses)
        pred_loss = loss_fn(labels, predictions)
        total_loss = pred_loss + regularization_loss

    # Update gradient
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update loss and accuracy
    loss.update_state(total_loss)
    accuracy.update_state(labels, predictions)


def test_one_batch(inputs, labels, model, loss_fn, loss, accuracy):
    predictions = model(inputs, training=False)
    regularization_loss = tf.math.add_n(model.losses)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss + regularization_loss

    # Update loss and accuracy
    loss.update_state(total_loss)
    accuracy.update_state(labels, predictions)


def main():
    ################################################## parse the arguments
    parser = argparse.ArgumentParser(description='Train retouch detection network.')
    parser.add_argument('--src_path', type=str,         default='./split_names', help='source path')
    parser.add_argument('--src_frac', type=float,       default=1.0, help='amount of training dataset')

    parser.add_argument('--method', type=str,           default="median", help='blur, median, noise or multi')
    parser.add_argument('--log_path', type=str,         default='logs', help='log path')

    parser.add_argument('--network', type=str,          default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet or XceptionNet')
    # parser.add_argument('--checkpoint_path', type=str, 	default="./logs/20200311_182113_noise_Total_91/checkpoint/weights_20", help='checkpoint path')
    parser.add_argument('--checkpoint_path', type=str,  default="", help='')
    parser.add_argument('--regularizer', type=float,    default=0.0001, help='regularizer')

    parser.add_argument('--epoch', type=int,            default=30, help='epoch')
    parser.add_argument('--batch_size', type=int,       default=16, help='batch size')
    parser.add_argument('--start_lr', type=float,       default=5e-5, help='start learning rate')
    parser.add_argument('--lr_update_rate', type=float, default=0.95, help='learning rate update rate')

    parser.add_argument('--debug', type=bool,           default=False, help='True or False')

    args = parser.parse_args()

    SRC_PATH        = args.src_path
    SRC_FRAC        = args.src_frac

    METHOD          = args.method
    LOG_PATH        = args.log_path

    NETWORK         = args.network
    CHECKPOINT_PATH = args.checkpoint_path
    REG             = args.regularizer

    EPOCHS          = args.epoch
    BATCH_SIZE      = args.batch_size
    START_LR        = args.start_lr
    LR_UPDATE_RATE  = args.lr_update_rate

    DEBUG           = args.debug

    print_args(args)


    ################################################## Create directories
    log_path, ckpt_path, history, train_summary_writer, valid_summary_writer = load_logFiles(LOG_PATH, METHOD, NETWORK)
    write_args(args, join(log_path, 'args.txt'))


    ################################################## Setup the training options
    # Load model
    NUM_CLASS = 4 if METHOD == "multi" else 2
    model = load_model(NETWORK, REG, NUM_CLASS)


    ################################################## Setup the dataset
    # Load data
    train_dataset, train_total = configure_dataset(  SRC_PATH, METHOD, 'train', BATCH_SIZE, \
                                        shuffle=True, repeat=False, frac=SRC_FRAC)
    test_dataset, test_total = configure_dataset(   SRC_PATH, METHOD, 'test', BATCH_SIZE, \
                                        shuffle=True, repeat=False, frac=SRC_FRAC)
    valid_dataset, valid_total = configure_dataset(  SRC_PATH, METHOD, 'valid', BATCH_SIZE, \
                                        shuffle=True, repeat=False, frac=SRC_FRAC)


    ################################################## Setup learning environments
    # optimizer and loss function
    lr_schdule = tf.keras.optimizers.schedules.ExponentialDecay(
        START_LR,
        decay_steps=int(train_total / BATCH_SIZE),
        decay_rate=LR_UPDATE_RATE,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schdule)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Load checkpoint
    if CHECKPOINT_PATH != "":
        model.load_weights(CHECKPOINT_PATH)


    ################################################## Train the model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    log_history(history, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)

    step = 1
    for epoch in range(EPOCHS):

        # Epoch initialization
        offset = 0
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # Train one epoch
        for inputs, labels in train_dataset:
            train_one_batch(inputs, labels,
                            model=model, loss_fn=loss_fn, optimizer=optimizer,
                            loss=train_loss, accuracy=train_accuracy)

            # Save tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=step)

            # Log batch information
            offset += inputs.shape[0]
            step += 1
            log_one_batch("Train", offset, train_total, train_loss, train_accuracy, history)

        # Save checkpoint
        model.save_weights(join(ckpt_path, f"weights_{epoch}"), save_format='h5')

        # Validation
        offset = 0
        for inputs, labels in valid_dataset:
            test_one_batch(inputs, labels,
                           model=model, loss_fn=loss_fn,
                           loss=valid_loss, accuracy=valid_accuracy)

            # Log batch information
            offset += inputs.shape[0]
            log_one_batch("Validation", offset, valid_total, valid_loss, valid_accuracy, history)

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=step)
            tf.summary.scalar('accuracy', valid_accuracy.result(), step=step)

        # Log epoch information
        log_history(history, f"[Epoch: {epoch:2d}] train_loss: {train_loss.result():.4f}, train_accuracy: {train_accuracy.result():.4f}%", 100)
        log_history(history, f"            valid_loss: {valid_loss.result():.4f}, valid_accuracy: {valid_accuracy.result():.4f}%\n", 100)
        log_history(history, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)

    # Test the model
    offset = 0
    for inputs, labels in test_dataset:
        test_one_batch(inputs, labels,
                       model=model, loss_fn=loss_fn,
                       loss=test_loss, accuracy=test_accuracy)

        # Log batch information
        offset += inputs.shape[0]
        log_one_batch("Test", offset, test_total, test_loss, test_accuracy, history)

    history.close()


if __name__ == "__main__":
    K.clear_session()
    main()


