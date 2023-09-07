import tensorflow as tf
import numpy as np 
from utils import *
def matrix_fact(U,V):


class ALS:
    def __init__(self, sparse_matrix, U, V):
        self.U = U
        self.V = V
        pass

    def train_step(self, num_iter = 1000, lr = 0.01, l2_reg):
        opt = tf.optimizers.Adam(lr)
        for i in range(num_iter):
            if i%2 ==0: 
                with tf.GradientTape() as tape:
                    loss = sparse_loss(sparse_matrix, self.U, self.V, l2_reg)
                gradients = tape.gradient(loss, [self.U])
                opt.apply_gradients(zip(gradient, [self.U]))
            else: 
                with tf.GradientTape() as tape:
                    loss = sparse_loss(sparse_matrix, self.U, self.V, l2_reg)
                gradients = tape.gradient(loss, [self.V])
                opt.apply_gradients(zip(gradient, [self.V]))
            if i % 99 ==0:
                print(f"Epoch {i+1}/{num_epochs}, Loss: {loss.numpy()}")



    def prediction(self, user_id = 10, k=6):
        scores = score(self.U[user_id])
        ordered_scores = np.argsort(scores)[::-1]
        top_k = ordered_scores[:k]

        return top_k