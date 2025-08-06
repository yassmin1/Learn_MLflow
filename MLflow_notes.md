# 🧠 MLflow Models & Model Registry

This document explains how to use **MLflow Models**, **Custom Python Models**, **Model Registry**, and **MLflow Projects** to manage and deploy machine learning models end-to-end.

---

## 📦 MLflow Models

**MLflow Models** standardize how ML models are saved, logged, and deployed.
They support multiple ML libraries using **Flavors** (e.g., scikit-learn, TensorFlow, PyTorch).

### ✅ Key Features

* Save models in a consistent format.
* Automatically log models using `mlflow.autolog()`.
* Store model metadata in an `MLmodel` YAML file (includes loader and environment info).

### 🛠️ Model API Functions

```python
mlflow.sklearn.save_model(model, path)        # Save to local filesystem
mlflow.sklearn.log_model(model, artifact_path) # Log to MLflow Tracking
mlflow.sklearn.load_model(uri)                # Load from path, run, or cloud
```

### 📂 Supported Load Locations

* Local folder: `./model`
* From run: `runs:/<run_id>/model`
* From registry: `models:/MyModel/Production`
* From cloud: S3, GCS, etc.

### 🔍 Tips

* Use `mlflow.last_active_run()` to get the latest run ID.
* Use `log_model(..., artifact_path='my_path')` for custom names (more control than `autolog()`).

---

## 🧩 Custom Python Models

MLflow allows you to log and serve **custom models** when built-in flavors don’t fit.

### Steps to Create Custom Model

1. **Inherit from `mlflow.pyfunc.PythonModel`**
2. Implement:

   * `load_context(self, context)`
   * `predict(self, context, model_input)`

### Example Skeleton

```python
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = ...
    
    def predict(self, context, input_data):
        return my_custom_prediction_logic(input_data)
```

### Saving/Loading Custom Models

```python
mlflow.pyfunc.save_model(path="my_model", python_model=MyModel())
mlflow.pyfunc.load_model("my_model")
```

### 🧪 Model Evaluation

Use `mlflow.evaluate()` to:

* Test model performance
* Log metrics (e.g., accuracy, F1)
* Visualize SHAP values in Tracking UI

---

## 🚀 Model Serving

MLflow lets you serve models as REST APIs.

### 🔧 Start Local Server

```bash
mlflow models serve -m "models:/MyModel/1" --port 5000
```

### 🌐 REST API Endpoints

| Endpoint       | Description                       |
| -------------- | --------------------------------- |
| `/ping`        | Service status                    |
| `/health`      | Health check                      |
| `/version`     | MLflow version info               |
| `/invocations` | Accepts data, returns predictions |

### 📨 Send a Request

```bash
curl -X POST http://127.0.0.1:5000/invocations \
     -H "Content-Type: application/json" \
     -d '{"columns":["feature1",...],"data":[[val1,...]]}'
```

* Input formats supported: CSV, JSON (dataframe\_split or dataframe\_records)

---

## 📘 Model Registry

MLflow **Model Registry** is a central place to:

* Track model versions
* Promote models (Staging → Production)
* Enable collaboration between teams

### Registering Models

```python
mlflow.register_model(model_uri="runs:/<run_id>/model", name="MyModel")
# or during training:
mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="MyModel")
```

### 🔢 Versioning Logic

* New version created each time you register under the same model name.
* MLflow auto-increments versions (v1 → v2 → v3...).

### 🖥️ Registry UI

* View models and versions
* Promote/demote stages
* Compare model performance

---

## 🏷️ Model Stages

| Stage        | Description                  |
| ------------ | ---------------------------- |
| `None`       | Model not yet assigned       |
| `Staging`    | Under evaluation/testing     |
| `Production` | Live and serving predictions |
| `Archived`   | Retired models               |

### 🔄 Changing Stage Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=2,
    stage="Production"
)
```

---

## 🔁 Model Deployment

### 2 Deployment Options:

1. **In Code**:

```python
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
predictions = model.predict(input_data)
```

2. **As REST API**:

```bash
mlflow models serve -m "models:/MyModel/Production" --port 5000
```

---

## 🧳 MLflow Projects

MLflow **Projects** package your ML code so it can be reused and shared.

### 🗂️ Project Structure

```
salary_model/
├── MLproject
├── train_model.py
├── python_env.yaml
└── requirements.txt
```

### 🔧 MLproject File Example

```yaml
name: salary_model

conda_env: python_env.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.01}
    command: "python train_model.py --lr {learning_rate}"
```

### 🧪 Key Benefits

* Run locally or remotely
* Reproducible environments
* Works well with MLflow Tracking & Registry

---

## 📎 Summary

MLflow enables you to:

✅ Log and manage models
✅ Create and serve custom models
✅ Evaluate and explain models
✅ Register and version models
✅ Package ML code for reuse and deployment

