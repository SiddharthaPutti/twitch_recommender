import pandas as pd 
import numpy as np 
import os

def load_data():
    cols = ["user","stream","streamer","start","stop"]
    df = pd.read_csv('100k_a.csv', header= None, names= cols)
    df.user = pd.factorize(df.user)[0]+1
    df['streamer_Raw'] = df.streamer
    df.streamer = pd.factorize(df.streamer)[0]+1
    # print("Num users: ", df.user.nunique())
    # print("Num streamers: ", df.streamer.nunique())
    # print("Num interactions: ", len(df))
    # print("Estimated watch time: ", (df['stop']-df['start']).sum() * 5 / 60.0)

if __name__ == "__main__":
    load_data()