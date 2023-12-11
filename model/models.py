from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def train_RF(X_train, y_train):
    print("-------------------")
    print("-- RANDOM FOREST --")
    print("-------------------")
    model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def train_NN(X_train, y_train):
    print("-------------------")
    print("- NEURAL  NETWORK -")
    print("-------------------")
    model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                          beta_1=0.9, beta_2=0.999, early_stopping=True,
                          epsilon=1e-08, hidden_layer_sizes=(128, 64),
                          learning_rate='constant', learning_rate_init=0.001,
                          max_iter=400, momentum=0.9, n_iter_no_change=10,
                          nesterovs_momentum=True, power_t=0.5, random_state=1,
                          shuffle=True, solver='lbfgs', tol=0.0001,
                          validation_fraction=0.1, verbose=False, warm_start=False)
    model.fit(X_train, y_train)
    return model


def train_SVC(X_train, y_train):
    print("-------------------")
    print("------- SVM -------")
    print("-------------------")
    model = SVC(kernel='linear', gamma='auto')
    model.fit(X_train, y_train)
    return model


def eval_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    result = classification_report(y_val, y_pred)
    accSVC = accuracy_score(y_val, y_pred)
    print(result)
    print("\nAccuracy: ", accSVC)
    return result, accSVC


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc