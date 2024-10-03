from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from zenml import pipeline
from pipelines.data_ingestion_pipe import data_ingestion_pipeline, database_generation_pipeline, auto_eda_plots
from pipelines.data_pre_processing_pipe import data_preprocessing_pipeline
from pipelines.machine_learning_pipe import machine_learning_pipeline
import uvicorn

# Create FastAPI app instance
app = FastAPI()

# Request body for pipeline input data
class PipelineInput(BaseModel):
    path: str = '../data/'              # Path to data directory
    meta_filename: str = 'meta_table_data.csv'  # Filename for the meta table
    database_path: str = '../data/database/database.csv'  # Path to the database file
    model_type: str = 'rfr'             # ML model to use, e.g., 'rfr' for RandomForest or 'abr' for AdaBoost

# Define the main ZenML pipeline as a function
@pipeline(enable_cache=False, name='main_pipeline_php')
def main_pipeline_php(path: str, meta_filename: str, database_path: str, model_type: str):
    try:
        # Ingesting raw experimental data and combining them
        data_ingestion = data_ingestion_pipeline(dir_path=path)

        # Selecting/filtering data, combining, and cleaning based on the experimental meta table
        database_generation = database_generation_pipeline(
            dir_path=path, 
            database=data_ingestion, 
            filename=meta_filename
        )

        # Auto generation of plots for selected thermal properties
        auto_eda = auto_eda_plots(
            dir_path=path, 
            database=database_generation
        )

        # Data pre-processing before ML
        data_ml = data_preprocessing_pipeline(
            data_path=database_path, 
            data=auto_eda
        )

        # Run machine learning based on the selected model type
        if model_type not in ['rfr', 'abr']:
            raise ValueError(f"Unsupported model type '{model_type}'. Use 'rfr' or 'abr'.")

        rmse, r2 = machine_learning_pipeline(data=data_ml, model_name=model_type)

        # Return model evaluation metrics
        return {
            "model_type": model_type,
            "rmse": rmse,
            "r2": r2
        }
    except Exception as e:
        raise RuntimeError(f"Pipeline execution error: {str(e)}")


# API endpoint to trigger the pipeline
@app.post("/run-pipeline/")
async def run_pipeline(pipeline_input: PipelineInput):
    try:
        # Unpack input data
        path = pipeline_input.path
        meta_filename = pipeline_input.meta_filename
        database_path = pipeline_input.database_path
        model_type = pipeline_input.model_type

        # Trigger the ZenML pipeline and return results
        results = main_pipeline_php(
            path=path, 
            meta_filename=meta_filename, 
            database_path=database_path, 
            model_type=model_type
        )

        return {"message": "Pipeline executed successfully!", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


# Optional: Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "OK", "message": "API is healthy"}


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
