# Class for BNN using Edward.
import tensorflow as tf
import edward as ed
from edward.models import Normal
import numpy as np
import time


class transition_model(object):
    def __init__(self, a_prev, s_prev, s_ph, n_neurons):
        # Note that the arguments above need not be placeholders.
        # Weights
        self.Wc_h_h = Normal(loc=tf.zeros([n_neurons, n_neurons]),
                             scale=0.1*tf.ones([n_neurons, n_neurons]))
        self.Wc_g_h = Normal(loc=tf.zeros([n_neurons, n_neurons]),
                             scale=0.1*tf.ones([n_neurons, n_neurons]))
        self.Bc_h_h = Normal(loc=tf.zeros([1, n_neurons]),
                             scale=tf.ones([1, n_neurons]))
        self.Bc_g_h = Normal(loc=tf.zeros([1, n_neurons]),
                             scale=tf.ones([1, n_neurons]))
        with tf.variable_scope('bnn'):
            self.qWc_g_h = Normal(loc=tf.Variable(
                tf.random_normal([n_neurons, n_neurons])),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([n_neurons, n_neurons]))))
            self.qBc_g_h = Normal(loc=tf.Variable(
                tf.random_normal([1, n_neurons])),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([1, n_neurons]))))
            self.qWc_h_h = Normal(loc=tf.Variable(
                tf.random_normal([n_neurons, n_neurons])),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([n_neurons, n_neurons]))))
            self.qBc_h_h = Normal(loc=tf.Variable(
                tf.random_normal([1, n_neurons])),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([1, n_neurons]))))

        # Placeholders
        self.a_prev = a_prev
        self.s_prev = s_prev
        self.s_ph = s_ph
        self.s = self._create_net()
        self.inference = ed.KLqp({self.Wc_h_h: self.qWc_h_h, self.Bc_h_h:
                                  self.qBc_h_h, self.Wc_g_h: self.qWc_g_h,
                                  self.Bc_g_h: self.qBc_g_h}, data={self.s:
                                                                    s_ph})
        self.inference.initialize(n_samples=10)
        weights = [self.qWc_g_h, self.qWc_g_h]
        self.stats = [[dist.mean(), dist.variance()] for dist in weights]
        self.sess = ed.get_session()

        # initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # n_epoch = 5
        # n_batch = 1
        # n-samples = 10
        # scale = 1 / 1
        # TODO: create a separate bulk updater
        # self.inference.initialize(n_iter=1 * n_epoch, n_samples=5,
        # scale={y: N / M})

        # old params
        self.mu_old_vals = None
        self.var_old_vals = None

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='bnn')
        self.saver = tf.train.Saver(train_vars)

        mus = tf.concat([i[0] for i in self.stats], 0)
        varss = tf.concat([i[1] for i in self.stats], 0)

        # private placeholders for prior statistics.
        self._prior_mu_placeholders = \
            [tf.placeholder(tf.float32, i[0].get_shape())
             for i in self.stats]
        self._prior_var_placeholders = \
            [tf.placeholder(tf.float32, i[1].get_shape())
             for i in self.stats]
        # mus_prior = tf.concat(prior_mu_placeholders, 0)
        # mus = tf.concat([i[0] for i in stats], 0)

        # Feed dictionary for KL calculation.
        self.feed_dict = {tensor: None for tensor in
                          self._prior_mu_placeholders}
        self.feed_dict.update({tensor: None for tensor in
                               self._prior_var_placeholders})

        mus_prior = tf.concat(self._prior_mu_placeholders, 0)
        vars_prior = tf.concat(self._prior_var_placeholders, 0)
        self._kl_tensor = transition_model.kl_div_p_q(mus, varss,
                                                      mus_prior, vars_prior)

    def _create_net(self):
        # same architecture as in RAM
        gw = tf.matmul(self.a_prev, self.Wc_g_h) + self.Bc_g_h
        hw = tf.matmul(self.s_prev, self.Wc_h_h) + self.Bc_h_h
        s = tf.nn.relu(tf.add(gw, hw))
        return s

    def save_old_params(self):
        statistics_old = self.sess.run(self.stats)
        mu_old_vals = [i[0] for i in statistics_old]
        var_old_vals = [i[1] for i in statistics_old]
        # update the feed dictionary
        i = 0
        for mu, var in (zip(mu_old_vals, var_old_vals)):
            self.feed_dict[self._prior_mu_placeholders[i]] = mu
            self.feed_dict[self._prior_var_placeholders[i]] = var
            i += 1

    def save_model(self, save_path="./t_zero_bnn.ckpt"):
        self.saver.save(self.sess, save_path)

    def load_model(self, save_path="./t_zero_bnn.ckpt"):
        self.saver.restore(self.sess, save_path)

    def get_kl_vals(self):
        vals = self.sess.run(self._kl_tensor, feed_dict=self.feed_dict)
        return vals

    @staticmethod
    def kl_div_p_q(p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian.
        (Grant: I stole this code from openai/vime/dynamics/bnn)"""
        numerator = tf.square(p_mean - q_mean) + \
            tf.square(p_std) - tf.square(q_std)
        denominator = 2 * tf.square(q_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + tf.log(q_std) - tf.log(p_std))

    def construct_int_reward(self, mems, glimpse_reps, r_intrinsic, start_time):
        """r_intrinsic : preallocated numpy array of zeros with
                         size batch_size x nGlimpses
        """
        batch_size, nGlimpses = r_intrinsic.shape
        self.save_model()
        self.save_old_params()
        feed_dict = {self.s_prev: None,
                     self.a_prev: None,
                     self.s_ph: None}
        t_prev = start_time
        durations = [0,0]
        for b in xrange(batch_size):
            for t in xrange(nGlimpses-1):
                # update the posterior
                feed_dict[self.s_prev] = mems[t][np.newaxis, b, :]
                feed_dict[self.a_prev] = glimpse_reps[t][np.newaxis, b, :]
                feed_dict[self.s_ph] = mems[t+1][np.newaxis, b, :]

                t_prev = time.time()
                for _ in xrange(5):
                    self.inference.update(feed_dict)
                t_post = time.time()
                durations[0] += t_post-t_prev

                t_prev = time.time()
                r_intrinsic[b, t] = self.get_kl_vals()
                t_post = time.time()
                durations[1] += t_post - t_prev
                # throw out the update.
                # TODO: just do manual tf.Variable assignment instead
                # of using checkpoints
                self.load_model()
        print 'timings'
        print durations
        return
