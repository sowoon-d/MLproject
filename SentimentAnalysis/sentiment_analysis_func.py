import pandas as pd
from konlpy.tag import Okt 
import re 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import yaml



def load_data():

    # conf 값 가져오기
    with open('conf.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    train_df = pd.read_excel(conf['data_path'])
    train_df = train_df[train_df['text'].notnull()]

    # print('train_df>>>>>>>>>>>>>>>>>>>>>>>>> ',len(train_df))
    # train_df = train_df[:100]
    # print('train_df>>>>>>>>>>>>>>>>>>>>>>>>> ',len(train_df))

    return train_df


def preprocessing(**context):
    train_df = context['task_instance'].xcom_pull(task_ids='load_data_task')
    # ‘ㄱ ~‘힣’까지의 문자를 제외한 나머지는 공백으로 치환, 영문: a-z| A-Z
    train_df['text'] = train_df['text'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
    text = train_df['text'] # 시리즈 객체로 저장
    score = train_df['score']
    
    context['task_instance'].xcom_push(key='text', value=text)
    context['task_instance'].xcom_push(key='score', value=text)
    # return text, score


def split_dataset(**context):
    text = context['task_instance'].xcom_pull(task_ids='preprocessing_task', key='text')
    score = context['task_instance'].xcom_pull(task_ids='preprocessing_task', key='score')
    
    # Train용 데이터셋과 Test용 데이터 셋 분리
    # 1. 예측력을 높이기 위해 수집된 데이터를 학습용과 테스트 용으로 분리하여 진행
    # 2. 보통 20~30%를 테스트용으로 분리해 두고 테스트
    train_x, test_x, train_y, test_y = train_test_split(text, score , test_size=0.2, random_state=0)

    context['task_instance'].xcom_push(key='train_x', value=train_x)
    context['task_instance'].xcom_push(key='test_x', value=test_x)
    context['task_instance'].xcom_push(key='train_y', value=train_y)
    context['task_instance'].xcom_push(key='test_y', value=test_y)
    # return train_x, test_x, train_y, test_y


def tokenization_vectorization(**context):
    train_x = context['task_instance'].xcom_pull(task_ids='split_dataset_task', key='train_x')

    okt = Okt()
    tfv = TfidfVectorizer(tokenizer=okt.morphs, ngram_range=(1,2), min_df=3, max_df=0.9)
    tfv.fit(train_x)
    tfv_train_x = tfv.transform(train_x)
    
    #Save vectorizer.vocabulary_
    pickle.dump(tfv.vocabulary_,open('./model/'+"tfv.pkl","wb"))

    context['task_instance'].xcom_push(key='tfv_train_x', value=tfv_train_x)
    # return tfv_train_x


def model_gridsearchCV(**context):
    tfv_train_x = context['task_instance'].xcom_pull(task_ids='tokenization_vectorization_task', key='tfv_train_x')
    train_y = context['task_instance'].xcom_pull(task_ids='split_dataset_task', key='train_y')

    # conf에서 models 가져오기
    with open('conf.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    models = conf['models']

    tscv = TimeSeriesSplit(n_splits=10)

    results_estimator = {}
    results_score = {}

    for model in models:
        # print(model['name'])
        est = eval(model['class'])()

        gsearch = GridSearchCV(est, cv=tscv, param_grid=model['params'], scoring='accuracy', verbose=1, n_jobs=1)
        if model['name'] == 'GNB':
            gsearch.fit(tfv_train_x.todense(), train_y)
        else:
            gsearch.fit(tfv_train_x, train_y)

        results_estimator[model['name']]= gsearch.best_estimator_
        results_score[model['name']]= gsearch.best_score_

    best_model_name = max(results_score,key=results_score.get)
    best_model = results_estimator[best_model_name]


    context['task_instance'].xcom_push(key='best_model', value=best_model)
    context['task_instance'].xcom_push(key='best_model_name', value=best_model_name)
    # return best_model, best_model_name


def save_model(**context):
    best_model = context['task_instance'].xcom_pull(task_ids='model_gridsearchCV_task', key='best_model')
    best_model_name = context['task_instance'].xcom_pull(task_ids='model_gridsearchCV_task', key='best_model_name')
    filename = best_model_name + '_'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'
    pickle.dump(best_model, open('./model/'+filename,'wb'))
    
    context['task_instance'].xcom_push(key='filename', value=filename)
    # return filename


def load_model(**context):
    filename = context['task_instance'].xcom_pull(task_ids='save_model_task', key='filename')
    loaded_model = pickle.load(open('./model/'+filename, 'rb'))

    context['task_instance'].xcom_push(key='loaded_model', value=loaded_model)
    # return loaded_model


def load_model_predict(**context):
    loaded_model = context['task_instance'].xcom_pull(task_ids='load_model_task', key='loaded_model')

    text = '딱히 대단한 재미도 감동도 없는데 ~! 너무 과대 평가된 영화 중 하나'

    #입력 텍스트에 대한 전처리 수행
    input_text = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(text)
    input_text = [" ".join(input_text)]
    # 입력 텍스트의 피처 벡터화
    okt = Okt()
    tfv = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open('./model/'+"tfv.pkl", "rb")))
    tfv.fit(input_text)
    st_tfidf = tfv.transform(input_text)
    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    st_predict = loaded_model.predict(st_tfidf)

    #예측 결과 출력
    if(st_predict == 0):
        print(text, '\n>> 예측 결과: 부정 감성')
    else :
        print(text, '\n>> 예측 결과: 긍정 감성')


