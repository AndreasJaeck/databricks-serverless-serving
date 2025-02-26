# Fortran MLflow Deployment Guide

This guide walks through the steps to wrap a Fortran model with MLflow and deploy it to X86 architecture.

## Prerequisites

- gfortran compiler
- virtualenv
- Python 3.8+
- NumPy
- pandas
- MLflow
- Databricks account (for Databricks deployment)

## Project Structure

```
fortran_mlflow_project/
├── simple_model.f90        # Fortran model source
├── Makefile                # Build automation
├── wrapper.py              # Python MLflow wrapper
├── test_model.py           # Testing script
└── README.md               # Documentation
```

## Step 1: Create the Fortran Model

Create a file named `simple_model.f90`:

```fortran
! simple_model.f90
subroutine simple_add(a, b, result)
    implicit none
    real, intent(in) :: a, b
    real, intent(out) :: result
    
    result = a + b
end subroutine simple_add
```

## Step 2: Create the Makefile

Create a `Makefile` for easier compilation:

```makefile
.PHONY: compile clean all test deploy

all: compile

compile:
	f2py -c simple_model.f90 -m fortran_model

clean:
	rm -rf fortran_model*.so fortran_model_dir fortran_model_mlflow __pycache__

test: compile
	python test_model.py

deploy: compile
	python wrapper.py
	mlflow models serve -m fortran_model_mlflow -p 5000
```

## Step 3: Build and Test

```bash

# Compile the Fortran model 
# make compile

# Wrap the Fortran model with MLFlow
# make build_mlflow_model

# Run tests (will compile and build MLFlow model)
make test
```

## Step 4: Deploy to target Environment

### Local Deployment
Deploy MLFlow model 
```bash
# Deploy locally
make deploy
```
Run curl in another terminal
```bash
# Curl locally
curl_test
```

### Databricks Deployment

1. Upload the compiled Fortran module (.so file) and wrapper to DBFS:

```bash
databricks fs cp fortran_model.cpython-*.so dbfs:/models/fortran_model/
databricks fs cp wrapper.py dbfs:/models/fortran_model/
```

2. Create a Databricks notebook with:

```python
# Databricks notebook
import sys
import os
import mlflow
import pandas as pd

# Copy the module to a local directory
dbutils.fs.cp("dbfs:/models/fortran_model/fortran_model.cpython-*.so", "file:/tmp/fortran_model.so")
os.makedirs("/tmp/fortran_model_dir", exist_ok=True)

# Set up the environment
sys.path.append("/tmp/fortran_model_dir")

# Run the wrapper to log the model
%run /dbfs/models/fortran_model/wrapper.py

# Register the model
model_name = "FortranModel"
client = mlflow.tracking.MlflowClient()
model_uri = f"runs:/{run.info.run_id}/fortran_model"
result = mlflow.register_model(model_uri, model_name)
```

3. Create a Model Serving Endpoint through Databricks UI or API

## Step 5: Inference Examples

### Python Client

```python
import pandas as pd
import requests
import json

# Test data
test_data = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [1.0, 2.0, 3.0]
})

# Prepare the request
headers = {"Content-Type": "application/json"}
data = test_data.to_dict(orient="split")
request_body = json.dumps({"dataframe_split": data})

# Make a prediction request
response = requests.post(
    "http://localhost:5000/invocations", 
    headers=headers, 
    data=request_body
)

# Print the result
print(response.text)
```

### Databricks API Client

```python
import os
import requests
import json
import pandas as pd

# Set up authentication
databricks_instance = "your-workspace.cloud.databricks.com"
token = "your-access-token"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test data
test_data = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [1.0, 2.0, 3.0]
})

# Prepare the request
endpoint_name = "FortranModel"
url = f"https://{databricks_instance}/api/2.0/serving-endpoints/{endpoint_name}/invocations"
data = test_data.to_dict(orient="split")
request_body = json.dumps({"dataframe_split": data})

# Make a prediction request
response = requests.post(url, headers=headers, data=request_body)

# Print the result
print(response.text)
```

## Troubleshooting

1. **Compilation errors**: Ensure gfortran is installed and in your PATH
2. **Import errors**: Check that the fortran_model.so file is in the correct directory
3. **Databricks errors**: Verify X86 cluster configuration and permissions
