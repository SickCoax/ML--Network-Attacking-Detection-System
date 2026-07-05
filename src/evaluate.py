from sklearn.metrics import f1_score

def score(model , X_test , y_test) :

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test , y_pred , average = "weighted")

    return y_pred , f1

