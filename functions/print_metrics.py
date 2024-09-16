from sklearn import metrics

def get_acc(y_train, y_test, predict_train, predict_test):
    acc_train = metrics.accuracy_score(y_train, predict_train)
    acc_test  = metrics.accuracy_score(y_test, predict_test)

    return acc_train, acc_test

def get_auc(y_train, y_test, proba_train, proba_test):
    auc_train = metrics.roc_auc_score(y_train, proba_train, multi_class='ovr')
    auc_test = metrics.roc_auc_score(y_test, proba_test, multi_class='ovr')

    return auc_train, auc_test

def get_kappa(y_train, y_test, predict_train, predict_test):
    kappa_train = metrics.cohen_kappa_score(y_train, predict_train)
    kappa_test = metrics.cohen_kappa_score(y_test, predict_test)

    return kappa_train, kappa_test

def get_precision(y_train, y_test, predict_train, predict_test):
    precision_train = metrics.precision_score(y_train, predict_train, average='macro')
    precision_test = metrics.precision_score(y_test, predict_test, average='macro')

    return precision_train, precision_test

def get_recall(y_train, y_test, predict_train, predict_test):
    recall_train = metrics.recall_score(y_train, predict_train, average='macro') 
    recall_test = metrics.recall_score(y_test, predict_test, average='macro') 

    return recall_train, recall_test

def print_acc(acc_train, acc_test):
    print(f'Acuracia base train: {round(acc_train * 100, 2)}%')
    print(f'Acuracia base test: {round(acc_test * 100, 2)}%')

def print_kappa(kappa_train, kappa_test):
    print(f'Kappa_score base train: {round(kappa_train * 100, 2)}%') 
    print(f'Kappa_score base test: {round(kappa_test * 100, 2)}%')

def print_auc(auc_train, auc_test):
    print(f'AUC base train: {round(auc_train * 100, 2)}%')
    print(f'AUC base test: {round(auc_test * 100, 2)}%')

def print_precision(precision_train, precision_test):
    print(f'Precision base train: {round(precision_train * 100, 2)}%')
    print(f'Precision base test: {round(precision_test * 100, 2)}%')

def print_recall(recall_train, recall_test):
    print(f'Recall base train: {round(recall_train * 100, 2)}%')
    print(f'Recall base test: {round(recall_test * 100, 2)}%')