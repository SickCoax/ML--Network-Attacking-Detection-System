from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def get_X_y(df) :

    df = df.drop(["id" , "label"] , axis=1)

    X = df.drop(["attack_cat"] , axis = 1)

    y = df[["attack_cat"]]

    oe = OrdinalEncoder()

    y_arr = oe.fit_transform(y)

    y_arr = y_arr.flatten()

    y_arr = y_arr.astype(int)

    y = pd.Series(y_arr)

    return X , y