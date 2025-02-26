# Databricks notebook source

# wrapper.py
import os
import shutil
import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec, DataType
import pandas as pd


# Define MLflow custom model class
class FortranModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def load_context(self, context):
        """Load the Fortran model when the MLflow model is loaded."""
        # Import the compiled Fortran module
        import sys
        sys.path.append(context.artifacts["fortran_model_dir"])
        import fortran_model
        self.model = fortran_model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            # Extract input values from DataFrame
            a_values = model_input["a"].values.astype(np.float32)
            b_values = model_input["b"].values.astype(np.float32)

            # Initialize results array
            results = np.zeros(len(a_values), dtype=np.float32)

            # Call Fortran function for each input pair
            for i in range(len(a_values)):
                # Function returns the result directly
                results[i] = self.model.simple_add(a_values[i], b_values[i])

            return pd.DataFrame({"result": results})
        else:
            raise TypeError("Input must be a pandas DataFrame")

# COMMAND ----------
# Define input and output schema using float64 to match pandas default
input_schema = Schema([
    ColSpec(DataType.double, "a"),  # float64 instead of float32
    ColSpec(DataType.double, "b")
])
output_schema = Schema([ColSpec(DataType.double, "result")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [1.0, 2.0, 3.0]
})

# Create a directory for the Fortran model
os.makedirs("fortran_model_dir", exist_ok=True)

# Copy the compiled Fortran module to the model directory
import glob
fortran_so_files = glob.glob("fortran_model*.so")
if not fortran_so_files:
    raise FileNotFoundError(
        "fortran_model*.so not found. Please compile it first with: f2py -c simple_model.f90 -m fortran_model")

for so_file in fortran_so_files:
    shutil.copy(so_file, "fortran_model_dir/")

# Modify the end of wrapper.py
with mlflow.start_run() as run:
    # Log the model to MLflow tracking
    mlflow.pyfunc.log_model(
        artifact_path="fortran_model",
        python_model=FortranModelWrapper(),
        artifacts={"fortran_model_dir": "fortran_model_dir"},
        signature=signature,
        input_example=input_example
    )

    # Get the path to the model in the MLflow tracking directory
    run_id = run.info.run_id
    model_path = os.path.join(mlflow.get_artifact_uri(), "fortran_model")

    print(f"Model logged with run_id: {run_id}")
    print(f"Model path: {model_path}")

    # Check if we're running on Databricks
    is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ

    if not is_databricks:
        # Local development: Create symlink or copy from MLflow artifacts
        test_model_path = "fortran_model_mlflow"
        if os.path.exists(test_model_path):
            shutil.rmtree(test_model_path)

        # The model_path is likely a file:// URI, so we need to extract the actual path
        if model_path.startswith("file://"):
            model_path = model_path[7:]

        # Copy the model to the expected test location
        shutil.copytree(model_path, test_model_path)
        print(f"Copied model to {test_model_path} for testing")
    else:
        # Databricks: Use MLflow client to download artifacts if needed
        print(f"Model logged to MLflow tracking server on Databricks with run_id: {run_id}")
