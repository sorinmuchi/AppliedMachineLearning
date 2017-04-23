from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 2, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


"""
To run from shell, run below command first:
(GOOGLE CLOUD)
cd /home/cmazzaanthony/Kal-El
source activate env
export PYTHONPATH="../cleverhans":$PYTHONPATH

(MAC)
export PYTHONPATH="../cleverhans":$PYTHONPATH

Libraries are from Cleverhans here: https://github.com/openai/cleverhans
"""

class AdversarialCNNFGSM:

    def __init__(self, train_size=None, test_size=None):
        self.session = self.set_up_sess()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.retrieve_mnist_dataset(train_size, test_size)
        self.x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.y = tf.placeholder(tf.float32, shape=(None, 10))
        self.epochs = 6
        self.batch_size = 128
        self.learning_rate = 0.1
        self.classes = 10 # 10 numbers in MNIST dataset
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.attacK_samples = 10
        self.train_params = {
            'nb_epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        self.eval_params = {'batch_size': 128}

    def set_up_sess(self):
        if keras.backend.image_dim_ordering() != 'tf':
            keras.backend.set_image_dim_ordering('tf')

        sess = tf.Session()
        keras.backend.set_session(sess)
        return sess

    def retrieve_mnist_dataset(self, training_size, test_size):
        X_train, Y_train, X_test, Y_test = data_mnist()

        return X_train[:training_size, :, :, :], \
               Y_train[:training_size, :], \
               X_test[:test_size, :, :, :], \
               Y_test[:test_size, :]

    def train(self):
        tf.set_random_seed(1234)

        if not hasattr(backend, "tf"):
            raise RuntimeError("This tutorial requires keras to be configured"
                               " to use the TensorFlow backend.")

        # Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'tf':
            keras.backend.set_image_dim_ordering('tf')

        # Create TF session and set as Keras backend session
        sess = tf.Session()
        keras.backend.set_session(sess)

        label_smooth = .1
        self.Y_train = self.Y_train.clip(label_smooth / 9., 1. - label_smooth)

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))

        # Define TF model graph
        model = cnn_model()
        predictions = model(x)
        print("Defined TensorFlow model graph.")

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions, self.X_test, self.Y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Train an MNIST model
        train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, predictions, self.X_train, self.Y_train,
                    evaluate=evaluate, args=train_params)

        # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
        adv_x = fgsm(x, predictions, eps=0.3)
        X_test_adv, = batch_eval(sess, [x], [adv_x], [self.X_test], args=self.eval_params)
        print("Generated {} adversarial examples using the Fast Gradient Sign Method. I will evaluate the "
              "model accuracy with adversarial examples in the validation set.".format(X_test_adv.shape[0]))

        # Evaluate the accuracy of the MNIST model on adversarial examples
        accuracy = model_eval(sess, x, y, predictions, X_test_adv, self.Y_test,
                              args=self.eval_params)
        print('Validation accuracy on adversarial examples: ' + str(accuracy))

        print("Repeating the process, using adversarial training")
        # Redefine TF model graph
        model_2 = cnn_model()
        predictions_2 = model_2(x) # returns tensors, not sure how to change proportion of examples
        adv_x_2 = fgsm(x, predictions_2, eps=0.3)
        predictions_2_adv = model_2(adv_x_2) # returns tensors, not sure how to change proportion of examples

        def evaluate_2():
            # Evaluate the accuracy of the adversarialy trained MNIST model on
            # legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_2, self.X_test, self.Y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

            # Evaluate the accuracy of the adversarially trained MNIST model on
            # adversarial examples
            accuracy_adv = model_eval(sess, x, y, predictions_2_adv, self.X_test,
                                      self.Y_test, args=eval_params)
            print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

        # Perform adversarial training
        model_train(sess, x, y, predictions_2, self.X_train, self.Y_train,
                    predictions_adv=predictions_2_adv, evaluate=evaluate_2,
                    args=train_params)

        # Load JSMA adversarial examples
        X_jsma = np.load("jsma/X_adv_out_jsma.npy")
        Y_jsma = np.load("jsma/Y_adv_out_jsma.npy")
        # X_jsma = np.load("src/jsma/X_adv_out_jsma.npy")
        # Y_jsma = np.load("src/jsma/Y_adv_out_jsma.npy")
        print("Loaded {0} JSMA adversarial examples to test".format(X_jsma.shape[0]))

        # Evaluate the accuracy of the MNIST model on JSMA adversarial examples
        accuracy = model_eval(sess, x, y, predictions, X_jsma, Y_jsma,
                              args=self.eval_params)

        print('Trained with {0} FGSM examples. Test accuracy on FGSM adversarial examples: {1}'.format(X_test_adv.shape[0], accuracy))

    def train_with_adversarial(self):
        adv_model = cnn_model()
        predictions = adv_model(self.x)
        adv_x = fgsm(self.x, predictions, eps=0.3)
        predictions_adv = adv_model(adv_x)

        model_train(self.session, self.x, self.y, predictions, self.X_train, self.Y_train,
                    predictions_adv=predictions_adv,
                    evaluate=self.evaluate(predictions, predictions_adv),
                    args=self.train_params)

    def clean_up(self):
        self.session.close()

if __name__ == "__main__":
    # adv_cnn_fgsm = AdversarialCNNFGSM(train_size=None, test_size=None)
    adv_cnn_fgsm = AdversarialCNNFGSM(train_size=500, test_size=100)
    predictions = adv_cnn_fgsm.train()
