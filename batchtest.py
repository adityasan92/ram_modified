import tensorflow as tf
import edward as ed
from edward.models import Normal
import numpy as np
ed.set_seed(42)

# KL terms between different times
# Save older model.

# use p as new, q as old
# TODO: how do we make this part of the symbolic graph? Use a
# placeholder for the old data!
def kl_div_p_q(p_mean, p_std, q_mean, q_std):
    """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian.
    (Grant: I stole this code from openai/vime/dynamics/bnn)"""
    numerator = tf.square(p_mean - q_mean) + \
        tf.square(p_std) - tf.square(q_std)
    print 'kl tensors'
    denominator = 2 * tf.square(q_std) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + tf.log(q_std) - tf.log(p_std))

def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.05, size=N)
  return x, y

N = 10000  # size of training data
M = 128    # batch size during training
D = 10    # number of features

w_true = np.ones(D) * 5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(235, w_true)

def neural_network(x):
    gh_in = tf.tanh(tf.matmul(x,w)+b)
    g, h_prev = tf.split(gh_in, 2,axis=1)

    gw = tf.matmul(g, Wc_g_h) + Bc_g_h
    hw = tf.matmul(h_prev, Wc_h_h) + Bc_h_h
    h_out = tf.tanh(tf.add(gw,hw))

    h = tf.matmul(h_out, w2) + b2

    return tf.reshape(h, [-1])

def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches

data = generator([X_train, y_train], M)

X = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

n_neurons = 256

w = Normal(loc=tf.zeros([D,2*n_neurons]), scale=tf.ones([D,2*n_neurons]))
b = Normal(loc=tf.zeros([1,2*n_neurons]), scale=tf.ones([1,2*n_neurons]))
#-----
Wc_h_h = Normal(loc=tf.zeros([n_neurons, n_neurons]), scale=0.1*tf.ones([n_neurons, n_neurons]))
Wc_g_h = Normal(loc=tf.zeros([n_neurons, n_neurons]), scale=0.1*tf.ones([n_neurons, n_neurons]))
Bc_h_h = Normal(loc=tf.zeros([1,n_neurons]), scale=tf.ones([1,n_neurons]))
Bc_g_h = Normal(loc=tf.zeros([1,n_neurons]), scale=tf.ones([1,n_neurons]))
#-----
w2 = Normal(loc=tf.zeros([n_neurons,1]), scale=tf.ones([n_neurons,1]))
b2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

qw = Normal(loc=tf.Variable(tf.random_normal([D,2*n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D,2*n_neurons]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1,2*n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1,2*n_neurons]))))
qWc_g_h = Normal(loc=tf.Variable(tf.random_normal([n_neurons,n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_neurons,n_neurons]))))
qBc_g_h = Normal(loc=tf.Variable(tf.random_normal([1,n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1,n_neurons]))))
qWc_h_h = Normal(loc=tf.Variable(tf.random_normal([n_neurons,n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_neurons,n_neurons]))))
qBc_h_h = Normal(loc=tf.Variable(tf.random_normal([1,n_neurons])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1,n_neurons]))))
qw2 = Normal(loc=tf.Variable(tf.random_normal([n_neurons,1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_neurons,1]))))
qb2 = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

weights = [qWc_g_h, qWc_h_h]
stats = [[dist.mean(),dist.variance()] for dist in weights]

y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(1))

n_batch = int(N / M)
n_epoch = 5

inference = ed.KLqp({w: qw, b: qb, Wc_h_h:qWc_h_h,Bc_h_h:qBc_h_h,Wc_g_h:qWc_g_h,Bc_g_h:qBc_g_h,w2:qw2,b2:qb2}, data={y: y_ph})
inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, scale={y: N / M})
tf.global_variables_initializer().run()

saver = tf.train.Saver()
sess = ed.get_session()

# Create a list of placeholders which we put the statistics into.
# shape is [[mu1,var1],[mu2,var2],...] and mu1,var1 etc have their 
# own predefined shapes as well.
prior_mu_placeholders = [tf.placeholder(tf.float32,i[0].get_shape()) for i in \
                         stats]
mus_prior = tf.concat(prior_mu_placeholders, 0)
mus = tf.concat([i[0] for i in stats], 0)
prior_var_placeholders = [tf.placeholder(tf.float32,i[1].get_shape()) for i in \
                          stats]
vars_prior = tf.concat(prior_var_placeholders, 0)
varss = tf.concat([i[1] for i in stats], 0)

feed_dict = {tensor:None for tensor in prior_mu_placeholders}
feed_dict.update({tensor:None for tensor in prior_var_placeholders})
kl_tensor = kl_div_p_q(mus,varss,mus_prior,vars_prior)

# there would be some loop here, in thru which the bulk updated posterior
# comes through
# save model here and the parameters
save_path = saver.save(sess,"./t_zero_bnn.ckpt")
statistics_old = sess.run(stats)
mu_old_vals = [i[0] for i in statistics_old]
var_old_vals = [i[1] for i in statistics_old]

# Put this information into the feed_dict
i = 0
for mu,var in (zip(mu_old_vals,var_old_vals)):
    feed_dict[prior_mu_placeholders[i]] = mu
    feed_dict[prior_var_placeholders[i]] = var
    i+=1

# put the statistics in a placeholder
for _ in range(inference.n_iter):
    # In ram, this will pull from the replay pool.
    X_batch, y_batch = next(data)

    #statistics_prevstep = sess.run(stats)
    info_dict = inference.update({X: X_batch, y_ph: y_batch})
    #statistics_new = sess.run(stats)

    # calculate Dkl here analytically.
    kl = kl_tensor.eval(feed_dict = feed_dict)
    print kl # This will increase as we train.

    # Revert to old parameters. (should have same parameters we saved above)
    inference.print_progress(info_dict)

# TODO: there actually is an assign method for tf.Variables....so in principle
# we can remove some overhead by using this over the approximating weight
# distributions variables instead of saving and restoring.
saver.restore(sess, "./t_zero_bnn.ckpt")

y_post = ed.copy(y, {w: qw, b: qb, Wc_h_h:qWc_h_h,Bc_h_h:qBc_h_h,Wc_g_h:qWc_g_h,Bc_g_h:qBc_g_h,w2:qw2,b2:qb2})

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))
