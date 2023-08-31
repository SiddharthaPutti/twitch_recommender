# matrix factorization 

import tensorflow as tf

# MSE
def sparse_loss(sparse_matrix, user_emb, streamer_emb):
    pred = tf.gather_nd(
        tf.matmul(user_emb, streamer_emb, transpose_b=True), sparse_matrix.indices
    )
    loss = tf.losses.mean_squared_error(sparse_matrix.values, predictions)
    return loss 

class colab_filtering():
    def __init__(self):
        pass

def build_model(data):
    train, test = split_data(data)

