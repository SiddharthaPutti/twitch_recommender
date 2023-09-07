

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


def split_data(df, hold_out = 0.1):
    test = df.sample(frac = hold_out, replace = False)
    train = df[~df.index.isin(test.index)]
    return train, test

def build_watch_sparse(df):
    sparse_matrix = tf.SparseTensor(
        indices = df[['useriD','StreamerName']].values,
        values = df['ts'].values,
        dense_shape = [np.max(df['useriD']), np.max(df['StreamerName'])]
    )
    return sparse_matrix

def score(self, U,  measure):
    scores = tf.reduce_sum(tf.multiply(self.V, U), axis = 1) / (tf.norm(self.V, axis = 1)* tf.norm(U))
    return scores

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
