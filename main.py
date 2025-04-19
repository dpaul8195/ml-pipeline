import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, make_scorer
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb

# Load
def load_config(config_path):
    with open(config_path, "r") as file:
        raw_content = file.read()
    
    if config_path.lower().endswith('.rtf'):
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(raw_content)
    else:
        text = raw_content
    
    return json.loads(text)

config = load_config("algoparams_from_ui.json.rtf")

df = pd.read_csv("iris.csv")

# Extract
target_info = config["design_state_data"]["target"]
prediction_type = target_info["prediction_type"]
target_column = target_info["target"]
feature_handling = config["design_state_data"].get("feature_handling", {})
reduction_config = config["design_state_data"].get("feature_reduction", {})
model_config = config["design_state_data"].get("algorithms", {})
hyperparam_config = config["design_state_data"].get("hyperparameters", {})

# Prepare features
numerical_features = []
categorical_features = []
text_features = []

for feature_name, conf in feature_handling.items():
    if conf.get("is_selected", False):
        if conf["feature_variable_type"] == "numerical":
            numerical_features.append(feature_name)
        elif conf["feature_variable_type"] == "categorical":
            categorical_features.append(feature_name)
        elif conf["feature_variable_type"] == "text":
            text_features.append(feature_name)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature reduction
def get_feature_reduction():
    method = reduction_config.get("feature_reduction_method", "No Reduction")
    
    if method == "Correlation with target":
        k = int(reduction_config.get("num_of_features_to_keep", len(numerical_features)))
        return SelectKBest(score_func=f_regression, k=k)
    elif method == "Tree-based":
        k = int(reduction_config.get("num_of_features_to_keep", len(numerical_features)))
        return SelectFromModel(
            RandomForestRegressor(
                n_estimators=int(reduction_config.get("num_of_trees", 10))),
            max_features=k
        )
    elif method == "Principal Component Analysis":
        k = int(reduction_config.get("num_of_features_to_keep", len(numerical_features)))
        return PCA(n_components=k)
    return 'passthrough'

X = df[numerical_features + categorical_features + text_features]
y = df[target_column]

if prediction_type == "Classification" and (y.dtype == 'object' or not y.dtype.kind in 'biufc'):
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameter mapping
def get_model_and_params(model_name, model_conf, prediction_type):
    params = {}

    # Skip classification models for regression problems and vice versa
    if prediction_type == "Regression" and model_name in ["LogisticRegression", "RandomForestClassifier", 
                                                         "GBTClassifier", "DecisionTreeClassifier", 
                                                         "SVC", "KNeighborsClassifier", "MLPClassifier"]:
        return None, None
        
    if prediction_type == "Classification" and model_name in ["LinearRegression", "RandomForestRegressor", 
                                                           "GBTRegressor", "DecisionTreeRegressor", 
                                                           "SVR", "KNeighborsRegressor", "MLPRegressor"]:
        return None, None

    def ensure_list(param):
        if param is None:
            return []
        if isinstance(param, (int, float, str, bool)):
            return [param]
        if isinstance(param, range):
            return list(param)
        if isinstance(param, (list, np.ndarray)):
            return param
        return [param]  # fallback for other types

    
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=42)
        params = {
            'model__n_estimators': list(range(model_conf.get("min_trees", 10), model_conf.get("max_trees", 100) + 1)),
            'model__max_depth': list(range(model_conf.get("min_depth", 5), model_conf.get("max_depth", 20) + 1))
        }
    
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=42)
        params = {
            'model__n_estimators': list(range(model_conf.get("min_trees", 10), model_conf.get("max_trees", 100) + 1)),
            'model__max_depth': list(range(model_conf.get("min_depth", 5), model_conf.get("max_depth", 20) + 1))
        }
    
    elif model_name == "GBTRegressor":
        model = GradientBoostingRegressor(random_state=42)
        params = {
            'model__n_estimators': model_conf.get("num_of_BoostingStages", [100]),
            'model__learning_rate': [0.01, 0.1, 0.5],
            'model__max_depth': list(range(model_conf.get("min_depth", 3), model_conf.get("max_depth", 10) + 1))
        }
    
    elif model_name == "GBTClassifier":
        model = GradientBoostingClassifier(random_state=42)
        params = {
            'model__n_estimators': model_conf.get("num_of_BoostingStages", [100]),
            'model__learning_rate': [0.01, 0.1, 0.5],
            'model__max_depth': list(range(model_conf.get("min_depth", 3), model_conf.get("max_depth", 10) + 1))
        }
    
    elif model_name == "LinearRegression":
        model = LinearRegression()
        params = {
            'model__fit_intercept': [True, False]
        }
    
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=42)
        params = {
            'model__C': np.logspace(-3, 3, 7),
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__solver': ['saga']
        }
    
    elif model_name == "RidgeRegression":
        model = Ridge(random_state=42)
        params = {
            'model__alpha': np.logspace(-3, 3, 7)
        }
    
    elif model_name == "LassoRegression":
        model = Lasso(random_state=42)
        params = {
            'model__alpha': np.logspace(-3, 3, 7)
        }
    
    elif model_name == "ElasticNetRegression":
        model = ElasticNet(random_state=42)
        params = {
            'model__alpha': np.logspace(-3, 3, 7),
            'model__l1_ratio': np.linspace(0, 1, 5)
        }
    
    elif model_name == "xg_boost":
        model = xgb.XGBRegressor() if prediction_type == "Regression" else xgb.XGBClassifier()
        params = {
            'model__n_estimators': model_conf.get("max_num_of_trees", [100]),
            'model__max_depth': model_conf.get("max_depth_of_tree", [3, 5, 7]),
            'model__learning_rate': model_conf.get("learningRate", [0.01, 0.1, 0.2])
        }
    
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=42)
        params = {
            'model__max_depth': list(range(model_conf.get("min_depth", 3), model_conf.get("max_depth", 15) + 1)),
            'model__min_samples_split': [2, 5, 10]
        }
    
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(random_state=42)
        params = {
            'model__max_depth': list(range(model_conf.get("min_depth", 3), model_conf.get("max_depth", 15) + 1)),
            'model__min_samples_split': [2, 5, 10]
        }
    
    elif model_name == "SVM":
        model = SVR() if prediction_type == "Regression" else SVC()
        params = {
            'model__C': model_conf.get("c_value", [0.1, 1, 10]),
            'model__kernel': ['linear', 'rbf']
        }
    
    elif model_name == "SGD":
        model = SGDRegressor() if prediction_type == "Regression" else SGDClassifier()
        params = {
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__penalty': ['l2', 'l1', 'elasticnet']
        }
    
    elif model_name == "KNN":
        model = KNeighborsRegressor() if prediction_type == "Regression" else KNeighborsClassifier()
        params = {
            'model__n_neighbors': model_conf.get("k_value", [3, 5, 7]),
            'model__weights': ['uniform', 'distance']
        }
    
    elif model_name == "extra_random_trees":
        model = ExtraTreesRegressor() if prediction_type == "Regression" else ExtraTreesClassifier()
        params = {
            'model__n_estimators': model_conf.get("num_of_trees", [100]),
            'model__max_depth': model_conf.get("max_depth", [None, 5, 10])
        }
    
    elif model_name == "neural_network":
        model = MLPRegressor() if prediction_type == "Regression" else MLPClassifier()
        params = {
            'model__hidden_layer_sizes': model_conf.get("hidden_layer_sizes", [(50,), (100,)]),
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001]
        }
    
    params = {k: ensure_list(v) for k, v in params.items()}
    params = {k: v for k, v in params.items() if len(v) > 0}
    
    return model, params

scoring = make_scorer(r2_score if prediction_type == "Regression" else accuracy_score)

for model_name, model_conf in model_config.items():
    if not model_conf.get("is_selected", False):
        continue
    
    print(f"\nTraining {model_name}")
    
    model, params = get_model_and_params(model_name, model_conf, prediction_type)

    if model is None:
        print(f"Skipping {model_name} with {prediction_type} problem type")
        continue
        
    if params is None or len(params) == 0:
        continue
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('reduction', get_feature_reduction()),
        ('model', model)
    ])
    
    grid = GridSearchCV(pipeline, params, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("Best Parameters:", grid.best_params_)
    
    if prediction_type == "Regression":
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))
    else:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall:", recall_score(y_test, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))