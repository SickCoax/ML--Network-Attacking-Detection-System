import os
import pandas as pd
from preprocessing import get_X_y
from train import train_model
from evaluate import score

csv_path = os.path.join(
    os.path.dirname(__file__) , 
    ".." , 
    "dataset" , "" \
    "UNSW_NB15_training-set.csv"
)
df_train = pd.read_csv(csv_path)
X_train , y_train = get_X_y(df_train)


csv_path = os.path.join(
    os.path.dirname(__file__) , 
    ".." , 
    "dataset" ,
    "UNSW_NB15_testing-set.csv"
)
df_test= pd.read_csv(csv_path)
X_test , y_test = get_X_y(df_test)


model = train_model(X_train , y_train)
y_pred , f1 = score(model , X_test , y_test)

print("Model Prediction : ")
print(y_pred)
print()
print(f"F1 Score : {f1}")