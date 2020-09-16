import tensorflow as tf
import numpy as np

NN1_NEURONS = 128
NN2_NEURONS = 64

class Actor:

    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, n_features])
        self.action = tf.placeholder(tf.int32, None)
        self.td_error = tf.placeholder(tf.float32, None)


        # 2-layer neural network for approximating the policy function
        l1 = tf.layers.dense(
            inputs=self.state,
            units=NN1_NEURONS,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1)
        )

        l2 = tf.layers.dense(
            inputs=l1,
            units=NN2_NEURONS,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.action_prob = tf.layers.dense(
            inputs=l2,
            units=n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1)
        )

        log_prob = tf.log(self.action_prob[0, self.action])
        self.expected_v = tf.reduce_mean(log_prob * self.td_error)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.expected_v)

    def update(self, state, action, td):
        """ Predicts the value for the given current state-action pair
            and performs backpropogation to update the policy network's weights

        Args:
            state (np.array): the current state
            action (np.array): the selected action for the current state
            td (np.array): temporal difference error

        Returns:
            np.array: the expected value for the given state-action pair
        """
        feed_dict = {
            self.state: state,
            self.action: action,
            self.td_error: td
        }
        _, expected_v = self.sess.run([self.train_op, self.expected_v], feed_dict)
        return expected_v

    def act(self, state):
        probs = self.sess.run(self.action_prob, 
            {
                self.state: state
            }
        )
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())