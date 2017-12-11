# Class for BNN using Edward.
import tensorflow as tf
import edward as ed
from edward.models import Normal
import numpy as np

SMALL_NUM = 1.e-10


class transition_model(object):
    def __init__(self, a_prev, s_prev, s_ph, n_neurons, worker_id,
                 mother=None):
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
        self.scope = 'bnn' + '/' + str(worker_id)
        with tf.variable_scope(self.scope):
            self.qWc_g_h = Normal(loc=tf.Variable(
                tf.random_normal([n_neurons, n_neurons]),name='qWc_g_h_mu'),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([n_neurons, n_neurons]),
                               name='qWc_g_h_var')))
            self.qBc_g_h = Normal(loc=tf.Variable(
                tf.random_normal([1, n_neurons]),name='qBc_g_h_mu'),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([1, n_neurons]),
                               name='qBc_g_h_var')))
            self.qWc_h_h = Normal(loc=tf.Variable(
                tf.random_normal([n_neurons, n_neurons]),name='qWc_h_h_mu'),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([n_neurons, n_neurons]),
                               name='qWc_h_h_var')))
            self.qBc_h_h = Normal(loc=tf.Variable(
                tf.random_normal([1, n_neurons]),name='qBc_h_h_mu'),
                scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([1, n_neurons]),
                               name='qBc_h_h_var')))

        # the mother bnn from that gets bulk posterior updates.
        # we need this to copy back to the original parameters
        self.mother = mother

        # Datastream
        self.a_prev = a_prev
        self.s_prev = s_prev
        self.s_ph = s_ph
        self.s = self._create_net()
        # self.inference = ed.KLqp({self.Wc_h_h: self.qWc_h_h, self.Bc_h_h:
                                  # self.qBc_h_h, self.Wc_g_h: self.qWc_g_h,
                                  # self.Bc_g_h: self.qBc_g_h}, data={self.s:
                                                                    # s_ph})
        weights = [self.qWc_h_h, self.qWc_g_h]
        self.stats = [[dist.mean(), dist.variance()] for dist in weights]
        biases = [self.qBc_h_h, self.qBc_g_h]
        self.stats_b = [[dist.mean(), dist.variance()] for dist in biases]
        #self.sess = ed.get_session()

        self.first_mean = self.stats[0][1][0,0]

        # initialize variables
        # TODO: is this needed?
        #init = tf.global_variables_initializer()
        #self.sess.run(init)

        # n_epoch = 5
        # n_batch = 1
        # n-samples = 10
        # scale = 1 / 1
        # TODO: create a separate bulk updater
        # self.inference.initialize(n_iter=1 * n_epoch, n_samples=5,
        # scale={y: N / M})

        # key_lookup = tf.GraphKeys.TRAINABLE_VARIABLES
        if mother is not None: # children-based functions
            #self.kl_to_mother_op = self.kl_to_mother()
            self.revert_to_mother_ops = self.construct_revert_to_mother_ops()
            # with tf.control_dependencies(self.revert_to_mother_ops):
                # #self.kl_to_mother_op = self.kl_to_mother()
            # i = tf.constant(0)
            # term = lambda i: tf.less(i,5)
            # def body(i):
                # weights = tf.trainable_variables(scope=self.scope)
                # with tf.variable_scope(self.scope):
                    # self.inference = ed.KLqp({self.Wc_h_h: self.qWc_h_h, self.Bc_h_h:
                                              # self.qBc_h_h, self.Wc_g_h: self.qWc_g_h,
                                              # self.Bc_g_h: self.qBc_g_h}, data={self.s:
                                                                                # s_ph})
                    # self.inference.initialize(n_samples=10, var_list=weights)
                # inf_ops = [self.inference.loss,
                                # self.inference.increment_t,
                               # self.inference.train]
                # with tf.control_dependencies(inf_ops):
                    # return i+1
            # self.glimpse_update = tf.while_loop(term, body, [i],
                                                # parallel_iterations=1)
            # TODO: we need to supply data?
            weights = tf.trainable_variables(scope=self.scope)
            with tf.variable_scope(self.scope + '/train'):
                self.inference = ed.KLqp({self.Wc_h_h: self.qWc_h_h, self.Bc_h_h:
                                          self.qBc_h_h, self.Wc_g_h: self.qWc_g_h,
                                          self.Bc_g_h: self.qBc_g_h}, data={self.s:
                                                                            s_ph})
                self.inference.initialize(n_samples=10, var_list=weights,
                                         debug=True)
            self.inf_ops = [self.inference.train,
                            self.inference.increment_t,
                            self.inference.loss]
            self.glimpse_update = self.inf_ops
            train_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=self.scope+'/train')
            self.init_train = tf.variables_initializer(train_var_list)
        else:
            # datadict = {self.s:s_ph,
            self.inference = ed.KLqp({self.Wc_h_h: self.qWc_h_h, self.Bc_h_h:
                                      self.qBc_h_h, self.Wc_g_h: self.qWc_g_h,
                                      self.Bc_g_h: self.qBc_g_h}, data={self.s:
                                                                        s_ph})
            # the user must initialize on their own.
            #self.inference.initialize(n_samples=10)
            #self.inf_ops = [self.inference.train,
                            #self.inference.increment_t,
                            #self.inference.loss]
            self.inf_ops = None


    def _create_net(self):
        # same architecture as in RAM
        gw = tf.matmul(self.a_prev, self.Wc_g_h) + self.Bc_g_h
        hw = tf.matmul(self.s_prev, self.Wc_h_h) + self.Bc_h_h
        s = tf.nn.relu(tf.add(gw, hw))
        return s

    # TODO: we dont actually need to revert to mother. we have
    # a bunch of BNNs.
    def construct_revert_to_mother_ops(self,weights_mother=None):
        # key_lookup = tf.GraphKeys.TRAINABLE_VARIABLES
        # weights = tf.get_collection(key_lookup, scope=self.scope)
        weights = tf.trainable_variables(scope=self.scope)
        if weights_mother is None:
            # weights_mother = tf.get_collection(key_lookup, scope=self.mother.scope)
            weights_mother = tf.trainable_variables(scope=self.mother.scope)
        # create assign_ops
        ops = []
        for varr, varr_mother in zip(weights, weights_mother):
            # ops.append(varr.assign(varr_mother))
            # ops.append(varr.assign(np.zeros(varr.get_shape().as_list())))
            ops.append(varr.assign(np.ones(varr.get_shape().as_list())))
        return ops

    #def save_model(self, save_path="./t_zero_bnn.ckpt"):
        #self.saver.save(self.sess, save_path)

    #def load_model(self, save_path="./t_zero_bnn.ckpt"):
        #self.saver.restore(self.sess, save_path)

    #def get_kl_vals(self):
        #vals = self.sess.run(self._kl_tensor, feed_dict=self.feed_dict)
        #return vals

    @staticmethod
    def kl_div_p_q(p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian.
        (Grant: I stole this code from openai/vime/dynamics/bnn)"""
        numerator = tf.square(p_mean - q_mean) + \
            tf.square(p_std) - tf.square(q_std)
        denominator = 2 * tf.square(q_std) + 1.e-8
        result = tf.reduce_sum(
            numerator / denominator + tf.log(q_std + SMALL_NUM) - \
            tf.log(p_std +SMALL_NUM))
        # return result
        return tf.clip_by_value(result,0,1000)

    def kl_to_mother(self):
        """Creates kl divergence tensor between mother and this child."""
        mus = tf.concat([i[0] for i in self.stats], 0)
        varss = tf.concat([i[1] for i in self.stats], 0)
        mus_mother = tf.concat([i[0] for i in self.mother.stats], 0)
        varss_mother = tf.concat([i[1] for i in self.mother.stats], 0)

        return transition_model.kl_div_p_q(mus, varss, mus_mother,
                                           varss_mother)

    def kl_with_self(self):
        """Creates kl divergence tensor between mother and this child."""
        mus = tf.concat([i[0] for i in self.stats], 0)
        varss = tf.concat([i[1] for i in self.stats], 0)

        return transition_model.kl_div_p_q(mus, varss, mus,
                                           varss)
