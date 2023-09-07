# twitch_recommender

## Datasets

|                    | Twitch 100k | Twitch full |
|--------------------|-------------|-------------|
| #Users             | 100k        | 15.5M       |
| #Items (streamers) | 162.6k      | 465k        |
| #Interactions      | 3M          | 124M        |
| #Timesteps (10min) | 6148        | 6148        |


Twitch Streamer Recommendations:

I'll dive into the technicals: 
* We have a matrix of (users, streamers, watch_hours), We need to create two different matrices namely U, V with random initialization.
* such that A ~ U*V (approximately equal). To achieve this we iteratively update U, and V by calculating the mean squared error.for which A is streaming hours. 
* That is
* ```python
  for _ in iterations:
    pred = U*V
    CALCULATE MSE(A, pred)
    update U,V with gradients  
  ```
* let's assume U represents user embedding matrix and V represents streamer embeddings.
* U with (num_users, embd_dim) and V with (num_streamers, embd_dim) , here embd_dim is a hyperparameter/coder_defined.
* once the training is done, we take a test useriD, say 234:
  *   ```python
      U[234] # index 234th row from user embedding
      # and the entire V (streamer embedding)
      # perform similarity measure between U and V, you can dot product or cosine similarity etc... 
      # you will get scores, then sort and select top K streamers
      ``` 
* code executions without airflow dags can be found matfact.ipynb file.
