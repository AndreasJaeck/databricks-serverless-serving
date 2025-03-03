# Databricks notebook source
# MAGIC %md
# MAGIC # Fortran Model MLflow Wrapper
# MAGIC
# MAGIC This notebook wraps a compiled Fortran model with MLflow to enable deployment and serving.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Compile a Fortran model using f2py
# MAGIC 2. Wrap it with MLflow's Python Function API
# MAGIC 3. Log the model to MLflow tracking
# MAGIC 4. Register it to the Model Registry
# MAGIC 5. Deploy it to a serving endpoint
# MAGIC
# MAGIC The example uses a simple Fortran function that adds two numbers, but the same approach can be used for more complex models.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC We need to install the following tools:
# MAGIC - **Meson**: A build system used to compile Fortran code
# MAGIC - **gfortran**: The GNU Fortran compiler
# MAGIC
# MAGIC These tools will allow us to compile our Fortran code into a shared library that can be called from Python.

# COMMAND ----------
%pip install meson
dbutils.library.restartPython()

# COMMAND ----------
%sh sudo apt-get install -y gfortran

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compile Fortran Model
# MAGIC
# MAGIC Now we'll compile our Fortran code into a Python-callable module using f2py (Fortran to Python interface generator).
# MAGIC
# MAGIC ### What is f2py?
# MAGIC f2py is part of NumPy and provides a connection between Python and Fortran languages. It automatically creates a Python extension module from Fortran code.
# MAGIC
# MAGIC ### Compilation Process
# MAGIC 1. The `f2py` command takes our Fortran source code (`simple_model.f90`) as input
# MAGIC 2. It generates a Python extension module named `fortran_model`
# MAGIC 3. This module can be imported in Python and the Fortran functions are available as Python functions
# MAGIC
# MAGIC ### Important Note
# MAGIC The compilation architecture must match the deployment environment. If you compile on an x86 machine but deploy to ARM architecture, the library won't work correctly.

# COMMAND ----------
%sh f2py -c simple_model.f90 -m fortran_model

# COMMAND ----------
%sh mkdir -p fortran_model_dir

# COMMAND ----------
%sh cp fortran_model*.so fortran_model_dir/

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Fortran Model Compilation
# MAGIC
# MAGIC Before proceeding with the MLflow integration, it's important to verify that our Fortran model compiled correctly and can be called from Python.
# MAGIC
# MAGIC ### Testing Process
# MAGIC 1. We import the compiled Fortran module into Python
# MAGIC 2. Call the `simple_add` function with test inputs
# MAGIC 3. Verify that the function returns the expected result
# MAGIC
# MAGIC ### Expected Behavior
# MAGIC - The function should add two floating-point numbers correctly
# MAGIC - If there are import errors, it likely means the compilation failed
# MAGIC - If the function returns incorrect results, there may be issues with the Fortran code or data type conversion

# COMMAND ----------
# Test the Fortran model directly after compilation
def test_fortran_direct():
    """Test the Fortran model directly after compilation."""
    try:
        import numpy as np
        import fortran_model
        a = np.float32(1.0)
        b = np.float32(1.0)
        result = fortran_model.simple_add(a, b)

        print(f"Direct Fortran test: {a} + {b} = {result}")
        assert abs(result - 2.0) < 1e-5, f"Expected 2.0, got {result}"
        print("Direct Fortran test passed!")
        return True
    except ImportError as e:
        print(f"Error importing Fortran module: {e}")
        print("Make sure to run 'f2py -c simple_model.f90 -m fortran_model' first")
        return False
    except Exception as e:
        print(f"Error testing Fortran model: {e}")
        return False

# Execute the test
fortran_test_result = test_fortran_direct()
if not fortran_test_result:
    print("Fortran model test failed. Please check the compilation output.")
    dbutils.notebook.exit("Fortran model test failed")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Import Libraries
# MAGIC
# MAGIC We need to import various Python libraries to work with MLflow:
# MAGIC
# MAGIC - **os**: For file path handling
# MAGIC - **mlflow**: Core MLflow functionality for tracking, logging and model management
# MAGIC - **numpy**: For numerical operations and data type conversion
# MAGIC - **pandas**: For data manipulation and structuring input/output
# MAGIC - **MLflow.models**: For model signature definition
# MAGIC - **MLflow.types**: For defining data schemas

# COMMAND ----------
import os
import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec, DataType
import pandas as pd

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define MLflow Model Wrapper
# MAGIC
# MAGIC To make our Fortran model compatible with MLflow, we need to create a wrapper class that:
# MAGIC
# MAGIC 1. **Extends mlflow.pyfunc.PythonModel**: This allows MLflow to handle our custom model
# MAGIC 2. **Implements load_context()**: Loads the Fortran model when the MLflow model is loaded
# MAGIC 3. **Implements predict()**: Handles input data processing, calls the Fortran function, and formats results
# MAGIC
# MAGIC ### The Wrapper Architecture
# MAGIC
# MAGIC - **Initialization**: Sets up the wrapper with empty model reference
# MAGIC - **Loading**: During deployment, MLflow calls load_context() to load the Fortran .so file
# MAGIC - **Prediction**: Takes pandas DataFrame input, processes it, calls Fortran functions, and returns results
# MAGIC
# MAGIC ### Data Flow
# MAGIC
# MAGIC 1. Input data arrives as a DataFrame with columns "a" and "b"
# MAGIC 2. Values are extracted and converted to numpy float32 arrays
# MAGIC 3. The Fortran function is called for each pair of values
# MAGIC 4. Results are collected and returned as a DataFrame

# COMMAND ----------
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
# MAGIC %md
# MAGIC ## Define Model Schema and Example Input
# MAGIC
# MAGIC MLflow models benefit from having well-defined schemas and examples. These provide:
# MAGIC
# MAGIC 1. **Type Safety**: Ensures inputs and outputs match expected data types
# MAGIC 2. **Documentation**: Helps users understand the model's expected inputs
# MAGIC 3. **Compatibility**: Enables model serving platforms to validate requests
# MAGIC
# MAGIC ### Schema Components
# MAGIC
# MAGIC - **Input Schema**: Defines the expected input columns and their data types
# MAGIC - **Output Schema**: Defines the structure and data types of the model's output
# MAGIC - **Signature**: Combines input and output schemas into a complete signature
# MAGIC
# MAGIC ### Input Example
# MAGIC
# MAGIC We provide an example input that:
# MAGIC - Shows the correct format (pandas DataFrame with columns "a" and "b")
# MAGIC - Demonstrates multiple rows of input
# MAGIC - Uses typical values the model might encounter

# COMMAND ----------
# Define input and output schema using float64 to match pandas default
input_schema = Schema([
    ColSpec(DataType.double, "a"),
    ColSpec(DataType.double, "b")
])
output_schema = Schema([ColSpec(DataType.double, "result")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [1.0, 2.0, 3.0]
})


# COMMAND ----------
# MAGIC %md
# MAGIC ## Log Model with MLflow
# MAGIC
# MAGIC Now we'll package and log our model to MLflow's tracking server. This process:
# MAGIC
# MAGIC 1. **Creates a Run**: Establishes a new MLflow run to track this model version
# MAGIC 2. **Logs the Model**: Saves the model, artifacts, and metadata to MLflow
# MAGIC 3. **Records Parameters**: Documents key information about the model
# MAGIC
# MAGIC ### What Gets Logged
# MAGIC
# MAGIC - **Python Wrapper**: Our FortranModelWrapper class
# MAGIC - **Artifacts**: The compiled Fortran .so file
# MAGIC - **Signature**: The input/output schema we defined
# MAGIC - **Example**: A sample input for documentation and testing
# MAGIC
# MAGIC ### MLflow Model Format
# MAGIC
# MAGIC The logged model will be stored in MLflow's model format, which includes:
# MAGIC - Model code and dependencies
# MAGIC - A conda environment specification
# MAGIC - Model metadata (signature, etc.)
# MAGIC - Custom artifacts (our Fortran .so file)

# COMMAND ----------
# Log the model to MLflow tracking
with mlflow.start_run() as run:
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
    print(f"Model logged to MLflow tracking server on Databricks with run_id: {run_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register Model in Unity Catalog
# MAGIC
# MAGIC After logging our model to the MLflow tracking server, we can register it in the Model Registry for easier management and deployment. The Model Registry provides:
# MAGIC
# MAGIC 1. **Version Control**: Track different versions of your model
# MAGIC 2. **Lifecycle Management**: Transition models through stages (Development, Staging, Production)
# MAGIC 3. **Lineage Tracking**: Connect models to their training runs and data
# MAGIC 4. **Deployment Management**: Simplify deployment to serving endpoints
# MAGIC
# MAGIC ### Registration Process
# MAGIC
# MAGIC - We specify a fully qualified name using Unity Catalog's three-level namespace (catalog.schema.model_name)
# MAGIC - The model from our current run is registered in this location
# MAGIC - A version number is automatically assigned (incremented from the previous version)
# MAGIC - The registration returns a ModelVersion object with metadata about the registered model

# COMMAND ----------
# Register the model in the Model Registry

catalog= "<catalog>"
db = "<db>"
model_name = f"{catalog}.{db}.fortran_model"
result = mlflow.register_model(f"runs:/{run_id}/fortran_model", model_name)
print(f"Model registered as: {model_name} version {result.version}")


# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Registered MLflow Model
# MAGIC
# MAGIC Now that we've registered our model, we should verify that it works correctly when loaded from the registry. This validation:
# MAGIC
# MAGIC 1. **Confirms Registration**: Ensures the model was registered correctly
# MAGIC 2. **Tests Functionality**: Verifies that the loaded model produces correct predictions
# MAGIC 3. **Validates Schema**: Ensures inputs and outputs match the expected format
# MAGIC
# MAGIC ### Testing Process
# MAGIC
# MAGIC - Load the model directly from the Model Registry using its fully qualified name
# MAGIC - Create test data with varied values
# MAGIC - Make predictions using the loaded model
# MAGIC - Compare the results to expected outputs (simple addition)
# MAGIC - Display detailed comparisons for transparency
# MAGIC
# MAGIC This test simulates how the model will behave when deployed to a serving endpoint.

# COMMAND ----------
# Test the registered MLflow model
def test_registered_model():
    """Test the registered MLflow model."""
    try:
        print(f"Testing registered model: {model_name}, version {result.version}")

        # Load the registered model
        registered_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{result.version}")

        # Create test data
        test_data = pd.DataFrame({
            "a": [1.0, 2.5, 3.7],
            "b": [4.0, 5.5, 6.3]
        })

        # Make predictions
        predictions = registered_model.predict(test_data)

        # Expected results (a + b)
        expected = [5.0, 8.0, 10.0]
        actual = predictions["result"].values

        # Display results
        results_df = pd.DataFrame({
            "a": test_data["a"],
            "b": test_data["b"],
            "predicted_sum": predictions["result"],
            "expected_sum": expected
        })
        print("Test results:")
        print(results_df)

        # Verify results
        for i, (exp, act) in enumerate(zip(expected, actual)):
            assert abs(exp - act) < 1e-5, f"At index {i}: Expected {exp}, got {act}"

        print("✅ Registered MLflow model test passed!")
        return True
    except Exception as e:
        print(f"❌ Error testing registered model: {e}")
        return False


# Run the test
registered_model_test_result = test_registered_model()
if not registered_model_test_result:
    print("Warning: Registered model test failed, but continuing with endpoint creation.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create Endpoint for Model Serving
# MAGIC
# MAGIC After successfully testing our registered model, we can deploy it to a model serving endpoint. Model serving endpoints provide:
# MAGIC
# MAGIC 1. **REST API Access**: Allow applications to call the model over HTTP
# MAGIC 2. **Scalability**: Handle varying load with automatic scaling
# MAGIC 3. **Monitoring**: Track performance, latency, and errors
# MAGIC 4. **Traffic Management**: Support traffic splitting between model versions
# MAGIC
# MAGIC ### Endpoint Configuration
# MAGIC
# MAGIC - **Name**: A unique identifier for the endpoint
# MAGIC - **Served Entities**: The model(s) to be served, including version and compute resources
# MAGIC - **Workload Size**: Compute resources allocated (Small, Medium, Large)
# MAGIC - **Scale to Zero**: Whether to scale to zero instances when idle (saves costs)
# MAGIC - **Traffic Config**: How to distribute traffic among different models
# MAGIC
# MAGIC ### Deployment Process
# MAGIC
# MAGIC 1. Set up the MLflow deployment client for Databricks
# MAGIC 2. Create an endpoint configuration with our model information
# MAGIC 3. Deploy the endpoint, which will start provisioning resources
# MAGIC 4. The endpoint will become available when the state changes to "READY"

# COMMAND ----------
# Create a serving endpoint for the model
import mlflow
from mlflow.deployments import get_deploy_client

# Define endpoint name based on model name
endpointname = f"fortran_model_endpoint"

# Set up the deployment client
mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

# COMMAND ----------
# Create the endpoint
endpoint = client.create_endpoint(
  name=f"{endpointname}",
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
                "served_model_name": f"fortran_model-1",
                "traffic_percentage": 100
            }
        ]
    }
  }
)

print(f"Endpoint '{endpointname}' created successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test the Serving Endpoint
# MAGIC
# MAGIC Finally, we'll test our deployed endpoint to ensure it's working correctly. This test:
# MAGIC
# MAGIC 1. **Waits for Readiness**: Endpoints take time to provision (up to 10 minutes)
# MAGIC 2. **Sends a Request**: Submits test data to the endpoint's REST API
# MAGIC 3. **Validates Response**: Ensures the endpoint returns correct predictions
# MAGIC
# MAGIC ### Request Format
# MAGIC
# MAGIC The endpoint expects data in the "dataframe_split" format:
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_split": {
# MAGIC     "columns": ["a", "b"],
# MAGIC     "data": [[1.0, 1.0], [2.0, 2.0], ...]
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### Response Format
# MAGIC
# MAGIC The endpoint returns predictions as a JSON response with the model's output. For our model, this will include the "result" column with the sums.
# MAGIC
# MAGIC ### Troubleshooting Tips
# MAGIC
# MAGIC - If the endpoint doesn't become ready, check Databricks logs for errors
# MAGIC - If requests fail, verify that your input format matches the model's expected schema
# MAGIC - For performance issues, consider adjusting the workload size or scaling configuration

# COMMAND ----------
# Test the serving endpoint
def test_endpoint():
    """Test the deployed endpoint."""
    try:
        import time
        print(f"Testing endpoint: {endpointname}")

        # Wait for endpoint to be ready - could take several minutes
        max_wait_sec = 600
        wait_interval_sec = 15
        elapsed_sec = 0

        print("Waiting for endpoint to be ready...")
        while elapsed_sec < max_wait_sec:
            endpoint_status = client.get_endpoint(endpointname)
            state = endpoint_status.get("state", {}).get("ready")
            if state == "READY":
                print(f"✅ Endpoint is ready after {elapsed_sec} seconds")
                break

            print(f"Endpoint status: {state}. Waiting {wait_interval_sec} seconds...")
            time.sleep(wait_interval_sec)
            elapsed_sec += wait_interval_sec

        if elapsed_sec >= max_wait_sec:
            print("⚠️ Timed out waiting for endpoint to be ready. Proceeding with test anyway.")

        # Create test data
        test_data = pd.DataFrame({
            "a": [1.0, 2.5, 3.7],
            "b": [4.0, 5.5, 6.3]
        })

        # Query the endpoint
        print("Sending request to endpoint...")
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

        # Expected results (a + b)
        expected = [5.0, 8.0, 10.0]

        # Extract actual results - structure depends on response format
        # Adjust if your endpoint response has a different structure
        if isinstance(response, dict) and "predictions" in response:
            actual = response["predictions"]
        else:
            actual = response

        # Display comparison
        results_df = pd.DataFrame({
            "a": test_data["a"],
            "b": test_data["b"],
            "expected_sum": expected
        })
        print("\nResults comparison:")
        print(results_df)
        print("\nEndpoint response:")
        print(actual)

        print("✅ Endpoint test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False


# Run the endpoint test
print("Testing the deployed endpoint...")
endpoint_test_result = test_endpoint()
if not endpoint_test_result:
    print("Warning: Endpoint test failed. Check endpoint status and logs.")
else:
    print("All tests completed successfully!")