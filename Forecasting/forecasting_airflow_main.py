from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.models.xcom_arg import XComArg
import pendulum
from functions import *
from datetime import timedelta
import SWP.MLproject.Forecasting.forecasting_func as ff


default_args = {
        'owner': 'airflow',
        'catchup': False,
        'execution_timeout': timedelta(hours=6),
        'depends_on_past': False,
}

kst = pendulum.timezone("Asia/Seoul")

# DAG 설정
dag = DAG(
    dag_id = 'forecasting',
    default_args = default_args,
    description = "forecasting, (swpark)",
    # schedule_intervfal = "* * * * *",
    start_date = days_ago(3),
    tags = ['minutes','forecasting']
    # ,
    # max_active_runs=3,
    # concurrency=1
)


# Tasks 

start = BashOperator(
    task_id='start_bash',
    bash_command='echo "start forecasting!"',
    dag=dag
)

load_data = PythonOperator(
    task_id='load_data',
    python_callable=ff.load_data,
    provide_context=True,   # Xcom 값 사용 시 True
    dag=dag
)

split_dataset = PythonOperator(
    task_id='split_dataset',
    python_callable=ff.split_dataset,
    provide_context=True,
    dag=dag
)

model_gridsearchCV = PythonOperator(
    task_id='model_gridsearchCV',
    python_callable=ff.model_gridsearchCV,
    provide_context=True,
    dag=dag
)

save_model = PythonOperator(
    task_id='save_model',
    python_callable=ff.save_model,
    provide_context=True,
    dag=dag
)

import_model_predict = PythonOperator(
    task_id='import_model_predict',
    python_callable=ff.import_model_predict,
    provide_context=True,
    dag=dag
)

complete = BashOperator(
    task_id = 'complete_bash',
    bash_command='echo "complete sentiment_analysis!"'
)

start >> load_data >> split_dataset >> model_gridsearchCV >> save_model >> import_model_predict >> complete











