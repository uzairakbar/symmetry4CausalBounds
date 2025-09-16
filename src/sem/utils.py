import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def select_best_degree(
        X, y, C,            # treatment, outcome, confounder variables
        max_degree: int=5,  # X polynomial features degree
        cv: int=5           # CV folds to pick degree to best explain data
    ) -> Tuple[int, LinearRegression]:
    best_degree = 1
    best_model = None
    best_score = -np.inf

    for degree in range(1, max_degree + 1):
        features = PolynomialFeatures(
            degree, include_bias=False
        )
        X_features = features.fit_transform(X)

        # concatenate X polynomial features and confounder C
        XC = np.hstack([
            X_features, C.reshape(-1, 1)
        ])

        model = LinearRegression()
        score = cross_val_score(
            model, XC, y, 
            cv=cv, scoring='neg_mean_squared_error'
        ).mean()

        if score > best_score:
            best_score = score
            best_degree = degree
            best_model = (model, features)

    return best_degree, best_model


def fit_ground_truth_f(
        X, y, C,            # treatment, outcome, confounder variables
        best_degree: int,   # X polynomial features degree
    ) -> Tuple[NDArray, PolynomialFeatures, float]:
    features = PolynomialFeatures(
        best_degree, include_bias=False
    )
    X_features = features.fit_transform(X)

    # fit full model with confounder C observed
    XC = np.hstack([
        X_features, C.reshape(-1, 1)
    ])
    model = LinearRegression().fit(XC, y)

    # extract epsilon coefficient of confounding noise in Y
    epsilon = model.coef_.flatten()[-1]
    y_deconfounded = y - epsilon * C

    # fit f(X) = y - epsilon * C
    f = LinearRegression().fit(
        X_features, y_deconfounded
    ).coef_.reshape(-1, 1)
    return f, features, epsilon