# test_model.py
import os
import sys
import numpy as np
import pandas as pd
import mlflow.pyfunc


def test_fortran_direct():
    """Test the Fortran model directly after compilation."""
    try:
        # Import the compiled Fortran module
        import fortran_model

        # Test inputs
        a = np.float32(1.0)
        b = np.float32(1.0)

        # Call the Fortran function - returns the result directly
        result = fortran_model.simple_add(a, b)

        print(f"Direct Fortran test: {a} + {b} = {result}")
        assert abs(result - 2.0) < 1e-5, f"Expected 2.0, got {result}"
        print("Direct Fortran test passed!")
    except ImportError as e:
        print(f"Error importing Fortran module: make test{e}")
        print("Make sure to run 'f2py -c simple_model.f90 -m fortran_model' first")
        sys.exit(1)


def test_mlflow_model():
    """Test the MLflow wrapped model if it exists."""
    model_path = "fortran_model_mlflow"

    if not os.path.exists(model_path):
        print(f"MLflow model not found at {model_path}")
        print("Run the wrapper.py script first to create the MLflow model")
        return

    try:
        # Load the model
        loaded_model = mlflow.pyfunc.load_model(model_path)

        # Create test data
        test_data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0]
        })

        # Make predictions
        predictions = loaded_model.predict(test_data)

        print("MLflow model test results:")
        print(pd.concat([test_data, predictions], axis=1))

        # Verify results
        expected = [2.0, 4.0, 6.0]
        actual = predictions["result"].values

        for i, (exp, act) in enumerate(zip(expected, actual)):
            assert abs(exp - act) < 1e-5, f"At index {i}: Expected {exp}, got {act}"

        print("MLflow model test passed!")
    except Exception as e:
        print(f"Error testing MLflow model: {e}")
        raise


if __name__ == "__main__":
    print("Testing Fortran model integration")
    print("-" * 40)

    # Test direct Fortran interface
    test_fortran_direct()

    print("-" * 40)

    # Test MLflow wrapped model
    test_mlflow_model()