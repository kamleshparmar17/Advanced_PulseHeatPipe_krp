# for data pre-processing for ML
import pandas as pd
import numpy as np
from zenml import step, pipeline
from typing_extensions import Annotated
from typing import Tuple, Any
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

@step
def save_to_csv(data:pd.DataFrame, filename:str)->None:
    data.to_csv(filename, index=False)
    return None

@step
def step_xy_split(df:pd.DataFrame)->Tuple[Annotated[pd.DataFrame, 'x-features'], 
                                     Annotated[pd.Series, 'y-target TR[K/W]']]:
    y_true = df.pop('TR[K/W]')
    y_true = pd.Series(y_true)
    return df, y_true

@step
def step_train_test_splitter(x:pd.DataFrame, y:pd.Series)->Tuple[Annotated[pd.DataFrame, 'x_train'],
                                                            Annotated[pd.DataFrame, 'x_test'],
                                                            Annotated[pd.Series, 'y_train'],
                                                            Annotated[pd.Series, 'y_test']]:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return x_train, x_test, y_train, y_test

@step
def step_model_selection(model_name:str)->Annotated[RegressorMixin, 'ML Regression Model Selection']:
    if model_name.lower() == "rfr":
        model = RandomForestRegressor()
    elif model_name.lower() == 'abr':
        model = AdaBoostRegressor()
    else:
        raise ValueError
    return model

@step
def step_model_training(model:RegressorMixin, 
                   x_train:pd.DataFrame, 
                   y_train:pd.Series)->Annotated[RegressorMixin, 'ML Model Training']:
    model.fit(x_train, y_train)
    return model

@step
def step_model_prediction(model:RegressorMixin, 
                     x_test:pd.DataFrame)->Annotated[pd.Series, 'ML Predictions']:
    predictions = model.predict(x_test)
    predictions = pd.Series(predictions)
    return predictions

@step
def step_model_evaluation_rmse(y_pred:pd.Series, y_test:pd.Series)->Annotated[float, 'RMSE']:
    score = mean_squared_error(y_pred=y_pred, y_true=y_test)
    print(f"RMSE: {score}")
    return score

@step
def step_model_evaluation_r2(y_pred:pd.Series, y_test:pd.Series)->Annotated[float, 'R2']:
    score = r2_score(y_pred=y_pred, y_true=y_test)
    print(f"R2: {score}")
    return score

@step
def step_model_evaluation_r2_adj(r2: float, data: pd.DataFrame) -> float:
    n = len(data)              # total samples
    p = data.shape[1] - 1      # features (exclude target)

    if n > p + 1:
        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        r2_adj = r2

    print(f"Adjusted R2: {r2_adj}")
    return r2_adj