import tensorflow as tf
import numpy as np

GAMMA = 0.9
NN1_NEURONS = 128
NN2_NEURONS = 64
class Critic:

    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [None, n_features])
        self.v_next = tf.placeholder(tf.float32, [None, 1])
        self.reward = tf.placeholder(tf.float32, None)


        # 2-layer neural network for approximating the value function
        l1 = tf.layers.dense(
            inputs=self.state,
            units=NN1_NEURONS,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )

        l2 = tf.layers.dense(
            inputs=l1,
            units=NN2_NEURONS,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )

        self.v = tf.layers.dense(
            inputs=l2,
            units=1,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )

        self.td_error = self.reward + GAMMA * self.v_next - self.v
        self.loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, state, reward, state_next):
        """ Predict the value of the next state and compute the temporal difference
            and performs backpropogation to update the value network's weights

        Args:
            state (np.array): the current state
            reward (np.array): the reward for the current state
            state_next (np.array): the next state

        Returns:
            np.array: temporal difference error
        """
        v_next = self.sess.run(self.v,
            {
                self.state: state_next
            }
        )
        td_error, _ = self.sess.run(
            [self.td_error, self.train_op],
            {
                self.state: state, 
                self.v_next: v_next, 
                self.reward: reward
            }
        )
        return td_error