import pickle
import pandas as pd
import pytest


my_dict2 = {'Age': 68,
 'RestingBP': 150,
 'Cholesterol': 195,
 'Oldpeak': 0.0,
 'FastingBS': 1,
 'MaxHR': 132,
 }

my_data= pd.DataFrame([my_dict2])

def test_predict():
    model = pickle.load(open('catboost_model-2.pkl', 'rb'))
    prediction = model.predict(my_data)
    assert prediction==1
