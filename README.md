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
* We have a matrix of A = (users, streamers, watch_hours), We need to create two different matrices namely U, V with random initialization.
* such that A ~ U*V (approximately equal). To achieve this we iteratively update U, and V by calculating the mean squared error.
* That is
* ```python
  for _ in iterations:
    pred = U*V
    CALCULATE MSE(A, pred)
    update U,V with gradients  
  ```
* let's assume U represents user embedding matrix and V represents streamer embeddings 
