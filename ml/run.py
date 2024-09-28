# to run all ML pipelines

from pipelines.data_ingestion_pipe import data_ingestion_pipeline, database_generation_pipeline, auto_eda_plots
from pipelines.data_pre_processing_pipe import data_preprocessing_pipeline
from pipelines.machine_learning_pipe import machine_learning_pipeline

from zenml import pipeline
import sys, os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pipeline(enable_cache=False, name='main_pipeline_php')
def main_pipeline_php(path:str = '../data/',
                      meta_table:str = 'meta_table_data.csv',
                      ml_model:str = 'rfr'):
    try:
        # ingesting raw experimental data and combining them all
        data_ingestion = data_ingestion_pipeline(dir_path=path)

        # with help of the experimental metal table selecting/filtering data, combining, and cleaning
        database_generation = database_generation_pipeline(dir_path=path, 
                                                        database=data_ingestion, 
                                                        filename=meta_table)

        # auto generation of plots for selected thermal properties using PyPulseHeatPipe
        auto_eda = auto_eda_plots(dir_path=path, 
                                database=database_generation)

        # data pre-processing before ML
        data_ml = data_preprocessing_pipeline(data_path='../data/database/database.csv',
                                            data=auto_eda)

        # ML training and evaluation of ML model
        rmse, r2 = machine_learning_pipeline(data=data_ml, model_name=ml_model)
        # # random forest regressor
        # rmse_rfr, r2_rfr = machine_learning_pipeline(data=data_ml, model_name=ml_model)
        # # ada boost regressors
        # rmse_abr, r2_abr = machine_learning_pipeline(data=data_ml, model_name='abr')  
        results = {"meta table": meta_table,
                "ML model": ml_model,
                "RMSE": rmse,
                "R2": r2}
        return results
    
    except Exception as err:
        raise RuntimeError(f"something went wrong with machine learning pipeline:\n {str(err)}")

# running main pipeline
if __name__ == "__main__":
    main_pipeline_php(path = '../data/',
                      meta_table = 'meta_table_data.csv',
                      ml_model = 'rfr')

