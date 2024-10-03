# ML Pipeline
from zenml import pipeline
from zenml.client import Client
import os, sys
import pandas as pd
client = Client()
from typing import Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.data_split_ml_steps import (step_model_evaluation_r2,
                                       step_model_evaluation_rmse,
                                       step_model_prediction,
                                       step_model_selection,
                                       step_model_training,
                                       step_train_test_splitter,
                                       step_xy_split,
                                       train_test_split)

@pipeline(enable_cache=False, name='Machine Learning')
def machine_learning_pipeline(data:pd.DataFrame, model_name:str)->Tuple[float, float]:
    x, y = step_xy_split(data)
    x_train, x_test, y_train, y_test = step_train_test_splitter(x, y)
    model = step_model_selection(model_name=model_name)
    trained_model = step_model_training(model=model,
                                      x_train=x_train,
                                      y_train=y_train)
    predictions = step_model_prediction(model=trained_model,
                                        x_test=x_test)
    rmse = step_model_evaluation_rmse(y_pred=predictions,
                                      y_test=y_test)
    r2 = step_model_evaluation_r2(y_pred=predictions,
                                      y_test=y_test)
    return rmse, r2

    
