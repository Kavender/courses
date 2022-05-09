

def train_test_split(X, y, test_pct):
    data = zip(X, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train) # magic unzip trick
    x_test, y_test = zip(*test)
    return x_train, y_train, x_test, y_test


def f1_score(tp, fp, fn, tn):
    "F1 is the harmonic mean of precision and recall."
    _precision = precition(tp, fp, fn, tn)
    _recall = recall(tp, fp, fn, tn)
    return 2 * _precision * _recall/(_precision + _recall)


def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    return tp / (tp + fn)
