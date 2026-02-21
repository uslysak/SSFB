import random


from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

from loss import DeepWalker, Clustering
from SeGNE_utils import gamma_incrementer, RandomWalker
from SeGNE_utils import batch_input_generator, batch_label_generator,epoch_printer


class SeGNE:
    """
        SeGNE: trains node embeddings from random walks using L = L_embed + γ * L_cluster.
    """

    def __init__(self, args, graph):

        self.args = args
        self.graph = graph


        self.walker = RandomWalker(self.graph, False, self.args.P, self.args.Q)
        self.walker.preprocess_transition_probs()
        self.walks, self.degrees = self.walker.simulate_walks(self.args.num_of_walks,
                                                              self.args.random_walk_length)
        self.nodes = [node for node in self.graph.nodes()]
        del self.walker
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks * self.vocab_size
        self._build()

    def _build(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)

            seed_nodes = getattr(self.args, "seed_nodes", None)

            self.cluster_layer = Clustering(self.args, seed_nodes=seed_nodes)

            self.gamma = tf.placeholder(tf.float32, shape=())
            self.step = tf.placeholder(tf.float32, shape=())

            self.loss_dw = self.walker_layer()
            self.loss_cluster = self.cluster_layer(self.walker_layer)
            self.loss = self.loss_dw + self.gamma * self.loss_cluster

            self.global_step = tf.Variable(0, trainable=False)
            self.lr = tf.train.polynomial_decay(
                self.args.initial_learning_rate,
                self.global_step,
                self.true_step_size,
                self.args.minimal_learning_rate,
                self.args.annealing_factor
            )

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, global_step=self.global_step
            )

            self.init = tf.global_variables_initializer()

    def _feed_dict(self, walk, step, gamma):
        if len(walk) < self.args.random_walk_length:
            return None

        x = batch_input_generator(walk, self.args.random_walk_length, self.args.window_size)
        y = batch_label_generator(walk, self.args.random_walk_length, self.args.window_size)

        return {
            self.walker_layer.train_inputs: x,
            self.walker_layer.train_labels: y,
            self.gamma: float(gamma),
            self.step: float(step),
        }

    def train(self):

        self.current_step = 0
        self.current_gamma = self.args.initial_gamma

        with tf.Session(graph=self.g) as session:
            session.run(self.init)
            print("Model Initialized.")
            for repetition in range(self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step,
                                                           self.args.initial_gamma,
                                                           self.args.final_gamma,
                                                           self.current_gamma,
                                                           self.true_step_size)

                    feed_dict = self._feed_dict(self.walks[self.current_step - 1],

                                                self.current_step,
                                                self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end - start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss / self.vocab_size
                self.final_embeddings = self.walker_layer.embedding_matrix.eval()

        return self.final_embeddings