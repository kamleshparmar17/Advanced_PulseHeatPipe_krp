# ML Pipeline
from zenml import pipeline
from zenml.client import Client
import os, sys
import pandas as pd
client = Client()
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.data_split_ml_steps import (
    step_model_evaluation_r2,
    step_model_evaluation_r2_adj,
    step_model_evaluation_rmse,
    step_model_prediction,
    step_model_selection,
    step_model_training,
    step_train_test_splitter,
    step_xy_split,
    save_to_csv
)


@pipeline(enable_cache=False, name='Machine Learning')
def machine_learning_pipeline(data: pd.DataFrame, model_name: str) -> Tuple[float, float, float, float, float, float]:

    # ---------------- Split Features & Target ----------------
    x, y = step_xy_split(data)

    # ---------------- Train-Test Split ----------------
    x_train, x_test, y_train, y_test = step_train_test_splitter(x, y)

    # ---------------- Save Data ----------------
    save_to_csv(data=x_train, filename='../data/ml_data/x_train.csv')
    save_to_csv(data=x_test, filename='../data/ml_data/x_test.csv')
    save_to_csv(data=y_train, filename='../data/ml_data/y_train.csv')
    save_to_csv(data=y_test, filename='../data/ml_data/y_test.csv')

    # ---------------- Model Selection ----------------
    model = step_model_selection(model_name=model_name)

    # ---------------- Model Training ----------------
    trained_model = step_model_training(
        model=model,
        x_train=x_train,
        y_train=y_train
    )

    # ---------------- TEST Prediction ----------------
    predictions_test = step_model_prediction(
        model=trained_model,
        x_test=x_test
    )

    # ---------------- TRAIN Prediction ----------------
    predictions_train = step_model_prediction(
        model=trained_model,
        x_test=x_train
    )

    # ---------------- Save Test Predictions ----------------
    save_to_csv(
        data=predictions_test,
        filename=f'../data/ml_data/predictions_{model_name}.csv'
    )

    # =========================================================
    # ---------------- TEST METRICS ----------------
    # =========================================================
    rmse_test = step_model_evaluation_rmse(
        y_pred=predictions_test,
        y_test=y_test
    )

    r2_test = step_model_evaluation_r2(
        y_pred=predictions_test,
        y_test=y_test
    )

    r2_adj_test = step_model_evaluation_r2_adj(
        r2=r2_test,
        data=data
    )

    # =========================================================
    # ---------------- TRAIN METRICS ----------------
    # =========================================================
    rmse_train = step_model_evaluation_rmse(
        y_pred=predictions_train,
        y_test=y_train
    )

    r2_train = step_model_evaluation_r2(
        y_pred=predictions_train,
        y_test=y_train
    )

    r2_adj_train = step_model_evaluation_r2_adj(
        r2=r2_train,
        data=data
    )

    # =========================================================
    # ---------------- RETURN ALL METRICS ----------------
    # =========================================================
    return (
        rmse_train, r2_train, r2_adj_train,
        rmse_test, r2_test, r2_adj_test
    )