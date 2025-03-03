# Fortran MLflow Deployment Guide

This guide demonstrates how to wrap a compiled Fortran model with MLflow to enable deployment and serving on Databricks.

## Overview

This project shows how to:

1. Compile a Fortran model using f2py
2. Wrap it with MLflow's Python Function API
3. Log the model to MLflow tracking
4. Register it to the Model Registry
5. Deploy it to a serving endpoint

The example uses a simple Fortran function that adds two numbers, but the same approach can be used for more complex models.

## Project Structure

```
mlflow_fortran/
├── simple_model.f90        # Fortran model source 
├── mlflow_fortran_wrapper.py  # Databricks notebook for deployment
├── README.md               # Documentation
```

## Prerequisites

- Databricks workspace with access to Unity Catalog
- gfortran compiler (installed via apt-get in the notebook)
- Python environment with:
  - MLflow
  - NumPy
  - pandas

## Step 1: The Fortran Model

The example uses a simple Fortran function that adds two numbers:

```fortran
! simple_model.f90
function simple_add(a, b) result(c)
    implicit none
    real, intent(in) :: a, b
    real :: c
    
    c = a + b
end function simple_add
```

## Step 2: Compiling and Testing the Fortran Model

The Fortran model is compiled using f2py, which is part of NumPy and provides a connection between Python and Fortran:

1. Install required dependencies:
   ```
   %pip install meson
   %sh sudo apt-get install -y gfortran
   ```

2. Compile the Fortran code:
   ```
   %sh f2py -c simple_model.f90 -m fortran_model
   ```

3. Create a directory for the model artifacts:
   ```
   %sh mkdir -p fortran_model_dir
   %sh cp fortran_model*.so fortran_model_dir/
   ```

4. Test the compiled model directly:
   ```python
   import numpy as np
   import fortran_model
   a = np.float32(1.0)
   b = np.float32(1.0)
   result = fortran_model.simple_add(a, b)
   print(f"Direct Fortran test: {a} + {b} = {result}")
   ```

## Step 3: Creating the MLflow Wrapper

To make the Fortran model compatible with MLflow, we create a wrapper class:

```python
class FortranModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def load_context(self, context):
        # Import the compiled Fortran module
        import sys
        sys.path.append(context.artifacts["fortran_model_dir"])
        import fortran_model
        self.model = fortran_model

    def predict(self, context, model_input):
        # Extract input values from DataFrame
        a_values = model_input["a"].values.astype(np.float32)
        b_values = model_input["b"].values.astype(np.float32)
        
        # Initialize results array
        results = np.zeros(len(a_values), dtype=np.float32)
        
        # Call Fortran function for each input pair
        for i in range(len(a_values)):
            results[i] = self.model.simple_add(a_values[i], b_values[i])
            
        return pd.DataFrame({"result": results})
```

## Step 4: Defining Model Schema and Example Input

Define the input and output schema for the model:

```python
input_schema = Schema([
    ColSpec(DataType.double, "a"),
    ColSpec(DataType.double, "b")
])
output_schema = Schema([ColSpec(DataType.double, "result")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [1.0, 2.0, 3.0]
})
```

## Step 5: Logging the Model to MLflow

Log the model to MLflow tracking:

```python
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="fortran_model",
        python_model=FortranModelWrapper(),
        artifacts={"fortran_model_dir": "fortran_model_dir"},
        signature=signature,
        input_example=input_example
    )
    run_id = run.info.run_id
```

## Step 6: Registering the Model in Unity Catalog

Register the model in the Unity Catalog for easier management and deployment:

```python
catalog = "<catalog>"
db = "<db>"
model_name = f"{catalog}.{db}.fortran_model"
result = mlflow.register_model(f"runs:/{run_id}/fortran_model", model_name)
```

## Step 7: Creating an Endpoint for Model Serving

Deploy the model to a serving endpoint:

```python
from mlflow.deployments import get_deploy_client

endpointname = "fortran_model_endpoint"
mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

endpoint = client.create_endpoint(
  name=endpointname,
  config={
    "served_entities": [
        {
            "entity_name": f"{catalog}.{db}.fortran_model",
            "entity_version": f"{result.version}",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": "fortran_model-1",
                "traffic_percentage": 100
            }
        ]
    }
  }
)
```

## Step 8: Testing the Endpoint

Test the deployed endpoint:

```python
def test_endpoint():
    # Wait for endpoint to be ready
    # ...
    
    # Query the endpoint
    response = client.predict(
        endpoint=endpointname,
        inputs={
            "dataframe_split": {
                "columns": ["a", "b"],
                "data": [
                    [1, 1],
                    [2, 2],
                    [3, 3]
                ]
            }
        }
    )
    
    # Process and validate response
    print("Response received from endpoint:")
    print(response)
```

## Making Predictions with the Endpoint

### Request Format

The endpoint expects data in the "dataframe_split" format:
```json
{
  "dataframe_split": {
    "columns": ["a", "b"],
    "data": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
  }
}
```

### Python Client

```python
import pandas as pd
import requests
import json
from mlflow.deployments import get_deploy_client

# Using the MLflow deployment client
client = get_deploy_client("databricks")
response = client.predict(
    endpoint="fortran_model_endpoint",
    inputs={
        "dataframe_split": {
            "columns": ["a", "b"],
            "data": [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0]
            ]
        }
    }
)
```

## Troubleshooting

1. **Compilation errors**: 
   - Ensure gfortran is installed
   - Check the f2py compilation command for errors

2. **Import errors**: 
   - Verify that the fortran_model.so file is in the correct directory
   - Check path settings in the load_context method

3. **Endpoint errors**:
   - Check endpoint status with `client.get_endpoint(endpointname)`
   - Ensure input data format matches the expected schema
   - For performance issues, consider adjusting the workload size

4. **Architecture compatibility**:
   - The compilation architecture must match the deployment environment
   - If you compile on an x86 machine but deploy to ARM architecture, the library won't work correctly