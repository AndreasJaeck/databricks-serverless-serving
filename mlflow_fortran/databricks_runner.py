# Databricks notebook source

# MAGIC %md
# MAGIC # Run Makefile Targets
# MAGIC This notebook will execute the `Makefile` targets sequentially.

# COMMAND ----------
%sh sudo apt-get install gfortran

# COMMAND ----------
%pip install meson
dbutils.library.restartPython()

# COMMAND ----------
#%sh make compile


# COMMAND ----------
#%sh make build_mlflow_model


# COMMAND ----------
#%sh make build_mlflow_model


# COMMAND ----------
%sh make test
