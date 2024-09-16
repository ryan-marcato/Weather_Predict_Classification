def get_predicts(pipeline, X_train, X_test):
    predict_train = pipeline.predict(X_train)
    predict_test = pipeline.predict(X_test)

    return predict_train, predict_test

def get_probas(pipeline, X_train, X_test):
    proba_train = pipeline.predict_proba(X_train)
    proba_test = pipeline.predict_proba(X_test)

    return proba_train, proba_test