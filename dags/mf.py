# matrix factorization 

import tensorflow as tf
from preprocessData import *


# def split_data(data):
#     pass

# MSE
def sparse_loss(sparse_matrix, user_emb, streamer_emb):
    pred = tf.reduce_sum(
        tf.gather(user_emb, sparse_matrix.indices[:,0]) *
        tf.gather(streamer_emb, sparse_matrix.indices[:, 1]),
        axis = 1
    )

    loss = tf.losses.mean_squared_error(sparse_matrix.values, pred)
    return loss 



class colab_filtering():
    def __init__(self, embeddings_vars, loss, metrics=None):
        self._loss = loss
        self._embeddings_vars = embeddings_vars
        self._embeddings = {k: None for k in embeddings_vars}
        self._metrics = metrics
        
    def embed(self):
        return self._embeddings

    def train_step(self,num_iter = 1000, lr = 3e-4, opt = tf.optimizers.SGD):
        optimizer = opt(lr)
        metric_values = [collections.defaultdict(list) for _ in self._metrics]

        for i in range(num_iter+1):
            with tf.GradientTape() as tape: 
                loss = self._loss

            gradients = tape.gradient(loss, self._embeddings.values())
            optimizer.apply_gradients(zip(gradients, self._embeddings.values()))

            results = self._metrics

            if (i%10 ==0) or i == num_iter:
                print("\r iteration %d:" % i + ", ".join(
                    ["%s=%f" % (k,v) for r in results for k,v in r.items()]
                ), end = '')
                for metric_values, result in zip(metric_values, results):
                    for k,v in result.items():
                        metric_values[k].append(v)

        for k,v in self._embeddings_vars.items():
            self._embeddings[k] = v.numpy()

        return results






def build_model(train, test, emb_dim = 3, init_stddev = 1):
    # train, test = split_data(data)

    # # removing indexes that are not present in teain data from test
    # train_combined = set(train['combined'])
    # mask = test['combined'].isin(train_combined)
    # filtered_test = test[mask]

    # train_sparse = build_watch_sparse(train)
    # test_sparse = build_watch_sparse(test)


    num_users, num_streamers = train_sparse.dense_shape

    U = tf.Variable(tf.random.normal(
        [num_users+!,emb_dim], stddev = init_stddev
    ))
    V = tf.Variable(tf.random.normal(
        [num_streamers+1, emb_dim], stddev = init_stddev
    ))

    train_loss = sparse_loss(train, U, V)
    test_loss = sparse_loss(test, U, V)

    metrics = {'train_loss': train_loss,
                'test_loss': test_loss}
    
    embeddings = {'user_id': U,
                    'streamer_Name': V}
    
    return colab_filtering(embeddings, train_loss, [metrics])

    



