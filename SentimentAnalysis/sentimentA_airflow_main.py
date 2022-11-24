from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.models.xcom_arg import XComArg
import pendulum
from functions import *
from datetime import timedelta
import SWP.MLproject.SentimentAnalysis.sentiment_analysis_func as sa


default_args = {
        'owner': 'airflow',
        'catchup': False,
        'execution_timeout': timedelta(hours=6),
        'depends_on_past': False,
}

kst = pendulum.timezone("Asia/Seoul")

# DAG 설정
dag = DAG(
    dag_id = 'sentiment_analysis',
    default_args = default_args,
    description = "sentiment_analysis, (swpark)",
    # schedule_intervfal = "* * * * *",
    start_date = days_ago(3),
    tags = ['minutes','sentiment_analysis']
    # ,
    # max_active_runs=3,
    # concurrency=1
)


# Tasks 

start = BashOperator(
    task_id='start_bash',
    bash_command='echo "start sentiment_analysis!"',
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=sa.load_data,
    provide_context=True,
    dag=dag
)

preprocessing_task = PythonOperator(
    task_id='preprocessing_task',
    python_callable=sa.preprocessing,
    provide_context=True,
    dag=dag
)

split_dataset_task = PythonOperator(
    task_id='split_dataset_task',
    python_callable=sa.split_dataset,
    provide_context=True,
    dag=dag
)

tokenization_vectorization_task = PythonOperator(
    task_id='tokenization_vectorization_task',
    python_callable=sa.tokenization_vectorization,
    provide_context=True,
    dag=dag
)

model_gridsearchCV_task = PythonOperator(
    task_id='model_gridsearchCV_task',
    python_callable=sa.model_gridsearchCV,
    provide_context=True,
    dag=dag
)

save_model_task = PythonOperator(
    task_id='save_model_task',
    python_callable=sa.save_model,
    provide_context=True,
    dag=dag
)

load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=sa.load_model,
    provide_context=True,
    dag=dag
)

load_model_predict_task = PythonOperator(
    task_id='load_model_predict_task',
    python_callable=sa.load_model_predict,
    provide_context=True,
    dag=dag
)

complete = BashOperator(
    task_id = 'complete_bash',
    bash_command='echo "complete sentiment_analysis!"'
)

start >> load_data_task >> preprocessing_task >> split_dataset_task >> tokenization_vectorization_task >> model_gridsearchCV_task >> save_model_task >> load_model_task >> load_model_predict_task >> complete











