# 🧠 Automated Machine Learning Pipeline

An end-to-end machine learning pipeline that automatically configures and trains models based on a JSON or RTF configuration file. Designed for both classification and regression tasks with support for over 17 algorithms, automatic preprocessing, feature selection, hyperparameter tuning, and evaluation.

---

## 🚀 Features

- ✅ **Automatic Model Selection** via configuration
- ⚙️ **Flexible Setup** using `.json` or `.rtf` config files
- 🔄 **Complete Pipeline**: Imputation, encoding, scaling, feature reduction, model training
- 📊 **Evaluation**: Automatically chooses metrics based on prediction type
- 🧩 **Modular Design**: Easily extend to new models or processing steps

---

## 🧠 Supported Algorithms

| Regression Models         | Classification Models         |
|--------------------------|-------------------------------|
| Linear Regression         | Logistic Regression           |
| Ridge, Lasso, ElasticNet  | Random Forest Classifier      |
| Random Forest Regressor   | Gradient Boosting Classifier  |
| Gradient Boosted Trees    | XGBoost Classifier            |
| XGBoost Regressor         | SVM Classifier                |
| SVM Regressor             | Neural Network Classifier     |
| KNN Regressor             | KNN Classifier                |
| Decision Tree Regressor   | Decision Tree Classifier      |
| Extra Trees Regressor     | Extra Trees Classifier        |
| SGD Regressor             | SGD Classifier                |
| Neural Network Regressor  | ...and more!                  |

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dpaul8195/ml-pipeline.git
cd ml-pipeline
pip install -r requirements.txt

## 🛠️ Usage

1. 📝 **Prepare Configuration File**  
   Create a config file named `config.json` or `config.rtf` following the supported schema (see example below).

2. 📂 **Place Dataset**  
   Put your dataset (e.g., CSV file) inside the `data/` directory.  
   Example: `data/iris.csv`

3. 🚀 **Run the Pipeline**  
   Execute the main script to start the automated ML process:

   ```bash
   python main.py

4. 📈 **View Results**
    The script will automatically print evaluation metrics based on the task type (classification or regression).

## 🧾 Configuration Example

Here’s a minimal working example of a configuration file for a regression task:

<details>
<summary>Click to expand JSON</summary>

```json
{
  "design_state_data": {
    "target": {
      "prediction_type": "Regression",
      "target": "sepal_length"
    },
    "feature_handling": {
      "sepal_width": {
        "is_selected": true,
        "feature_variable_type": "numerical"
      },
      "species": {
        "is_selected": true,
        "feature_variable_type": "categorical"
      }
    },
    "feature_reduction": {
      "feature_reduction_method": "Tree-based",
      "num_of_features_to_keep": 3,
      "num_of_trees": 50
    },
    "algorithms": {
      "RandomForestRegressor": {
        "is_selected": true,
        "min_trees": 10,
        "max_trees": 100,
        "min_depth": 3,
        "max_depth": 10
      }
    }
  }
}

## 🧩 Configuration Options

| Section            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `target`           | Target column and prediction type (`Regression` or `Classification`)       |
| `feature_handling` | Select features to include and specify their types                          |
| `feature_reduction`| Optional feature reduction method (`Tree-based`, `PCA`, etc.)               |
| `algorithms`       | Choose models and set their hyperparameters                                 |
| `hyperparameters`  | Optional section for global GridSearch settings                             |

## 📁 Project Structure

```bash
ml-pipeline/
├── data                   
├── main.py                 
├── requirements.txt        
└── README.md            


---

## 🧪 Customization

To add new models or preprocessing steps:

- Extend the `get_model_and_params()` function in `main.py`
- Add corresponding entries in the configuration file
- Implement any required preprocessing logic in the pipeline

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss the idea.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ❤️ Credits

Created by [Debabrata Paul](https://github.com/dpaul8195)
