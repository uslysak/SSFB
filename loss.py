import math
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

"""
DeepWalker:
- Implements Node2vec style node embeddings.

Clustering:
- Adds a latent-space clustering regularizer.

"""
class DeepWalker:


    def __init__(self, args, vocab_size, degrees):

        self.args = args
        self.vocab_size = vocab_size
        self.degrees = degrees
        self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])

        self.train_inputs = tf.placeholder(tf.int64, shape=[None])

        self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                              -0.1 / self.args.dimensions,
                                                              0.1 / self.args.dimensions))

        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                                           stddev=1.0 / math.sqrt(self.args.dimensions)))

        self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size],
                                                        -0.1 / self.args.dimensions,
                                                        0.1 / self.args.dimensions))

    def __call__(self):

        self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
        self.input_ones = tf.ones_like(self.train_labels)
        self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs, [-1, 1])), [-1])
        self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix,
                                                        self.train_inputs_flat,
                                                        max_norm=1)

        self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes=self.train_labels_flat,
                                                             num_true=1,
                                                             num_sampled=self.args.negative_sample_number,
                                                             unique=True,
                                                             range_max=self.vocab_size,
                                                             distortion=self.args.distortion,
                                                             unigrams=self.degrees)

        self.embedding_losses = tf.nn.sampled_softmax_loss(weights=self.nce_weights,
                                                           biases=self.nce_biases,
                                                           labels=self.train_labels_flat,
                                                           inputs=self.embedding_partial,
                                                           num_true=1,
                                                           num_sampled=self.args.negative_sample_number,
                                                           num_classes=self.vocab_size,
                                                           sampled_values=self.sampler)

        return tf.reduce_mean(self.embedding_losses)


class Clustering:


    def __init__(self, args, seed_nodes=None):
        self.args = args

        self.seed_nodes = seed_nodes


        self.cluster_means = tf.Variable(
            tf.random_uniform(
                [self.args.cluster_number, self.args.dimensions],
                -0.1 / self.args.dimensions,
                0.1 / self.args.dimensions
            )
        )

    def __call__(self, Walker):

        if self.seed_nodes is not None:

            seed_ids = tf.constant(self.seed_nodes, dtype=tf.int32)

            centers = tf.nn.embedding_lookup(Walker.embedding_matrix, seed_ids)

        diffs = tf.expand_dims(Walker.embedding_partial, 1) - centers
        sq_dists = tf.reduce_sum(tf.square(diffs), axis=2)
        self.to_be_averaged = tf.reduce_min(sq_dists, axis=1)
        return tf.reduce_mean(self.to_be_averaged)