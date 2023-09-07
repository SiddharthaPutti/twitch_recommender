# matrix factorization 
import pandas as pd 
import numpy as np 
import tensorflow as tf
from preprocessData import *
tf.config.run_functions_eagerly(True)

# def split_data(data):
#     pass

# MSE
def sparse_loss(sparse_matrix, user_emb, streamer_emb, l2_reg = 0):
    # tf.gather - from the original dataframe take respective column and and from that column values consider them as indiced of embeddings to matmul. 
    # for example: sparse_matrix.indices[:,0] which is useriD's, take useriD as index of user_emb. 
    # similarly for streameriD. for a corresponding user and streamer embedding now multiply them. 
    pred = tf.reduce_sum(
        tf.gather(user_emb, sparse_matrix.indices[:,0]) *
        tf.gather(streamer_emb, sparse_matrix.indices[:, 1]),
        axis = 1
    )

    loss = tf.losses.mean_squared_error(sparse_matrix.values, pred)
    user_reg = l2_reg * tf.reduce_Sum(tf.square(user_emb))
    streamer_reg = l2_reg * tf.reduce_Sum(tf.square(streamer_emb)) 
    return loss + user_reg + streamer_reg


def embed_init(train, emb_dim = 3, init_stddev = 1):

    num_users, num_streamers = train.shape
    xavier_init = tf.initializers.GlorotNormal()

    U = tf.Variable(xavier_init(
        shape = (num_users+1,emb_dim)
    ))
    V = tf.Variable(xavier_init(
        shape = (num_streamers+1, emb_dim)
    ))

    return U,V

DOT= "dot"
COSINE = 'cosine'


class colab_filtering:
    def __init__(self, train_sparse,test_sparse,  U, V):
        self.train_sparse = train_sparse
        self.test_sparse = test_sparse
        self.U = U
        self.V =V

    def score(self, U,  measure):
        scores = tf.reduce_sum(tf.multiply(self.V, U), axis = 1) / (tf.norm(self.V, axis = 1)* tf.norm(U))
        return scores

    def train_step(self,num_iter = 1000, lr = 3e-4, l2_reg = 0.01):

        optimizer = tf.keras.optimizers.Adam(lr)

        for i in range(num_iter):
            with tf.GradientTape() as tape: 
                loss = sparse_loss(self.train_sparse, self.U,self.V, l2_reg)

            gradients = tape.gradient(loss, [self.U,self.V])
            optimizer.apply_gradients(zip(gradients, [self.U,self.V]))

    def prediction(self, user_id = 10, k=6):
        scores = self.score(self.U[user_id])
        ordered_scores = np.argsort(scores)[::-1]
        top_k = ordered_scores[:k]

        return top_k


def build_mf_model(train, test):
    U,V = embed_init(train)

    model = colab_filtering(train, test, U, V)
    model.train_step()
    rec = model.prediction(user_id= 512)
    print(rec)








    



