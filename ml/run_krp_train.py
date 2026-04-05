# to run all ML pipelines

from pipelines.data_ingestion_pipe import (
    data_ingestion_pipeline,
    database_generation_pipeline,
    auto_eda_plots
)
from pipelines.data_pre_processing_pipe import data_preprocessing_pipeline
from pipelines.machine_learning_pipe_train import machine_learning_pipeline

from zenml import pipeline
import sys, os


@pipeline(enable_cache=False, name='main_pipeline_php')
def main_pipeline_php(path: str = '../data/'):

    # =========================================================
    # ---------------- DATA INGESTION ----------------
    # =========================================================
    data_ingestion = data_ingestion_pipeline(dir_path=path)

    # =========================================================
    # ---------------- DATABASE GENERATION ----------------
    # =========================================================
    database_generation = database_generation_pipeline(
        dir_path=path,
        database=data_ingestion,
        filename='meta_table_data_5.csv'
    )

    # =========================================================
    # ---------------- AUTO EDA ----------------
    # =========================================================
    auto_eda = auto_eda_plots(
        dir_path=path,
        database=database_generation
    )

    # =========================================================
    # ---------------- DATA PREPROCESSING ----------------
    # =========================================================
    data_ml = data_preprocessing_pipeline(
        data_path='../data/database/database.csv',
        data=auto_eda
    )

    # =========================================================
    # ---------------- RANDOM FOREST ----------------
    # =========================================================
    (
        rmse_train_rfr, r2_train_rfr, r2_adj_train_rfr,
        rmse_test_rfr, r2_test_rfr, r2_adj_test_rfr
    ) = machine_learning_pipeline(
        data=data_ml,
        model_name='rfr'
    )

    # =========================================================
    # ---------------- ADABOOST ----------------
    # =========================================================
    (
        rmse_train_abr, r2_train_abr, r2_adj_train_abr,
        rmse_test_abr, r2_test_abr, r2_adj_test_abr
    ) = machine_learning_pipeline(
        data=data_ml,
        model_name='abr'
    )

    # =========================================================
    # ---------------- PRINT RESULTS ----------------
    # =========================================================

    def get_value(x):
        # Try to extract the value from StepArtifact or similar wrappers
        for attr in ('read', 'get', 'value', 'data', 'artifact', 'output', 'result'):
            if hasattr(x, attr):
                method = getattr(x, attr)
                try:
                    # If it's callable (method), call it
                    return method() if callable(method) else method
                except Exception:
                    continue
        return x

    def get_float(x):
        try:
            return float(get_value(x))
        except Exception:
            return 0.0  # fallback to 0.0 if conversion fails

    print("\n================ RANDOM FOREST RESULTS ================")
    print(f"Train -> RMSE: {get_float(rmse_train_rfr):.6f}, R²: {get_float(r2_train_rfr):.6f}, Adjusted R²: {get_float(r2_adj_train_rfr):.6f}")
    print(f"Test  -> RMSE: {get_float(rmse_test_rfr):.6f}, R²: {get_float(r2_test_rfr):.6f}, Adjusted R²: {get_float(r2_adj_test_rfr):.6f}")

    print("\n================ ADABOOST RESULTS =====================")
    print(f"Train -> RMSE: {get_float(rmse_train_abr):.6f}, R²: {get_float(r2_train_abr):.6f}, Adjusted R²: {get_float(r2_adj_train_abr):.6f}")
    print(f"Test  -> RMSE: {get_float(rmse_test_abr):.6f}, R²: {get_float(r2_test_abr):.6f}, Adjusted R²: {get_float(r2_adj_test_abr):.6f}")

    print("\n================ PIPELINE SUMMARY =====================")
    print("{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "RMSE_Tr", "R2_Tr", "AdjR2_Tr", "RMSE_Te", "R2_Te", "AdjR2_Te"))
    print("{:<12} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
        "RandomForest",
        get_float(rmse_train_rfr), get_float(r2_train_rfr), get_float(r2_adj_train_rfr),
        get_float(rmse_test_rfr), get_float(r2_test_rfr), get_float(r2_adj_test_rfr)))
    print("{:<12} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
        "AdaBoost",
        get_float(rmse_train_abr), get_float(r2_train_abr), get_float(r2_adj_train_abr),
        get_float(rmse_test_abr), get_float(r2_test_abr), get_float(r2_adj_test_abr)))

# =========================================================
# ---------------- RUN PIPELINE ----------------
# =========================================================
if __name__ == "__main__":
    main_pipeline_php()