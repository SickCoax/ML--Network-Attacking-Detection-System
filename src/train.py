from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

def train_model(X_train , y_train) :

    cat_cols = X_train.select_dtypes(include = ["object" , "string"]).columns
    num_cols = X_train.select_dtypes(include = ["number"]).columns

    weight = compute_sample_weight(
        y = y_train , 
        class_weight = "balanced"
    )

    preprocess = ColumnTransformer([
        ("cat" , OrdinalEncoder(handle_unknown = "use_encoded_value" , unknown_value = -1) , cat_cols ) ,
        ("num" , "passthrough" , num_cols)
    ])

    model = Pipeline([
        ("preprocess" , preprocess) , 
        ("xgbc" , XGBClassifier(
            random_state = 42 , 
            subsample = 0.8 , 
            colsample_bytree = 0.8 ,
            n_jobs = -1 ,
            max_depth = 8 , 
            min_child_weight = 4 ,
            n_estimators = 365 ,
            gamma = 1.1979 ,
            reg_lambda = 12.5152 ,
            reg_alpha = 3.0128 ,
            learning_rate = 0.3522
        ))
    ])

    # Hyperparamater Tunning is done in jupyter notebook 

    model.fit(
        X_train , 
        y_train , 
        xgbc__sample_weight = weight
    )

    return model