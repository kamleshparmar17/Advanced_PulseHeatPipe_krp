from fastapi import FastAPI
from enum import Enum
from run import main_pipeline_php

# class for ML model selection
class ModelSelection(str, Enum):
    RandomForestRegressor = 'rfr'
    AdaBoostRegressor = 'abr'

app = FastAPI(title="Pulsating Heat Pipe - Advanced Data Analytics and Machine Learning",
              description="""Web browser base interface to interact and trigger advanced analytics and machine learning pipelines for PHP experimental data.

website - www.nirmalparmar.info \n
mail - nirmalparmarphd@gmail.com

    Author - Dr. Nirmal Parmar, PhD
""")

@app.get("/models/{experiment_name}",
         description="Advanced data analysis and machine learning interface")
async def trigger_ml_pipeline(experiment_name: str = 'PHP 2',
                              experiment_description: str | None=None,
                              ml_model: ModelSelection = 'Random Forest Regressor',
                               experimental_data_path: str = "../data/",
                               meta_data_filename: str = "meta_table_data.csv",
                               ):
    """
    Advanced analysis and ML trigger
    """
    results = main_pipeline_php(path = experimental_data_path,
                                meta_table = meta_data_filename,
                                ml_model = ml_model,
                                experiment_name = experiment_name,
                                description = experiment_description)
    return results

# run this in terminal
# fastapi dev main.py