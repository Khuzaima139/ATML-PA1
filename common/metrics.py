from sklearn.metrics import accuracy_score, f1_score

def acc_f1(y_true, y_pred):
    return {
        "acc": 100 * accuracy_score(y_true, y_pred),
        "macro_f1": 100 * f1_score(y_true, y_pred, average="macro"),
    }