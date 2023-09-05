import pandas as pd 
import numpy as np
import os 
from datetime import datetime
from airflow.models import DAG, Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.operators.postgres_operator import PostgresOperator
from preprocessData import preproc
from mf import *
import random

def preprocess_df():
    batch_size = 200000
    pg_hook = PostgresHook(postgres_conn_id = 'postgresql_rec')
    offset = 0
    df = []

    while True: 
        query = f"SELECT * FROM dataframe LIMIT {batch_size} OFFSET {offset}"
        records = pg_hook.get_records(sql= query)

        if not records: 
            break

        df.extend(records)
        offset+= batch_size
    
    return preproc(random.shuffle(df))

def build_model(**kwargs):
    ti = kwargs['task_instance']
    train, test = ti.xcom_pull(task_ids = 'preprocess_df', key='key_name')
    build_mf_model(train, test)

with DAG(
    dag_id = 'first_af_dag',
    schedule_interval = '* * * * *',
    start_date = datetime(year= 2023, month = 8, day =24),
    catchup = False
) as dag:

    create_db = PostgresOperator(
        task_id = 'create_table',
        postgres_conn_id = "postgresql_rec",
        sql = "SQL/schema.sql",
    )

    # update_db = PostgresOperator(
    #     task_id = 'update_table',
    #     postgres_conn_id = "postgresql_rec",
    #     sql = "SQL/loadDf.sql",
    # )

with DAG(
    dag_id = 'preprocess_dag',
    schedule_interval = '* * * * *',
    start_date = datetime(year= 2023, month = 8, day =31),
    catchup = False
) as dag: 

    preprocess_data = PythonOperator(
        task_id = 'preprocess_df',
        python_callable = preprocess_df,
        do_xcom_push = True
    )


with DAG (
    dag_id = 'SGD_dag',
    schedule_interval = '* * * * *',
    start_date = datetime(year= 2023, month = 8, day =31),
    catchup = False
) as dag: 
    SGDregresor = PythonOperator(
        task_id = 'SGD',
        python_callable = build_model,
        do_xcom_push = True
    )
