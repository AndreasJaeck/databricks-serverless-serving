.PHONY: compile clean all test deploy databricks

all: compile

compile:
	@echo "Compiling Fortran model..."
	f2py -c simple_model.f90 -m fortran_model
	@echo "Done!"

fortran_model_dir: compile
	@echo "Creating Fortran model directory..."
	mkdir -p fortran_model_dir
	cp fortran_model*.so fortran_model_dir/
	@echo "Done!"

clean:
	@echo "Cleaning up..."
	rm -rf fortran_model*.so fortran_model_dir fortran_model_mlflow __pycache__
	@echo "Done!"

build_mlflow_model: clean fortran_model_dir
	@echo "Building MLflow model..."
	python wrapper.py
	@echo "Done!"

test: build_mlflow_model
	@echo "Running tests..."
	python test_model.py

deploy: test
	@echo "Deploying model locally..."
	mlflow models serve -m fortran_model_mlflow -p 5000

curl_test:
	@echo "Testing the deployed model with curl..."
	curl http://127.0.0.1:5000/invocations -H "Content-Type:application/json" \
		--data '{"dataframe_split": {"columns": ["a", "b"], "data": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]}}'

	@echo "\nDone!"
