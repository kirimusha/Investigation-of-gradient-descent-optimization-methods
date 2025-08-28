from sklearn.tree import DecisionTreeClassifier
import numpy as np

def adaboost(X, y, T=50):
    N = len(y)
    D = np.ones(N) / N
    classifiers = []
    alphas = []

    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y, sample_weight=D)
        pred = clf.predict(X)
        error = np.sum(D * (pred != y))

        if error >= 0.5:
            break

    alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
    D = D * np.exp(-alpha * y * pred)
    D = D / np.sum(D)

    classifiers.append(clf)
    alphas.append(alpha)

    def final_model(x):
        result = sum(alpha * clf.predict(x.reshape(1, -1))[0] for alpha, clf in zip(alphas, classifiers))
        return np.sign(result)

    return final_model