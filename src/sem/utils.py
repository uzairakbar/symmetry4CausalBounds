import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def select_best_degree(
    X,
    y,
    C,  # treatment, outcome, confounder variables
    max_degree: int = 5,  # X polynomial features degree
    cv: int = 5,  # CV folds to pick degree to best explain data
) -> tuple[int, LinearRegression]:
    best_degree = 1
    best_model = None
    best_score = -np.inf

    for degree in range(1, max_degree + 1):
        features = PolynomialFeatures(degree, include_bias=False)
        X_features = features.fit_transform(X)

        # concatenate X polynomial features and confounder C
        XC = np.hstack([X_features, C.reshape(-1, 1)])

        model = LinearRegression()
        score = cross_val_score(model, XC, y, cv=cv, scoring="neg_mean_squared_error").mean()

        if score > best_score:
            best_score = score
            best_degree = degree
            best_model = (model, features)

    return best_degree, best_model


def fit_ground_truth_f(
    X,
    y,
    C,  # treatment, outcome, confounder variables
    best_degree: int,  # X polynomial features degree
) -> tuple[NDArray, float, PolynomialFeatures, float]:
    """(W, b, features, epsilon) for the ground truth f(x) = phi(x) W + b.

    The INTERCEPT is returned, not discarded. Asm. 1's base clause closes the
    hypothesis class under constant shifts, and Lem. 2's set lives on the slice
    H_X = {h : E[h(X)] = E[Y]}; a ground truth fitted without a free intercept is
    not on that slice and is therefore not in the identified set the solver
    searches. `phi` here excludes the bias column (`include_bias=False`) and, for
    degree >= 2, its squared terms have nonzero mean even on centred X -- so the
    intercept is not a formality: dropping it moved h_* off the slice by 0.333 on
    the published optical experiment, more than eight standard errors of the level.
    """
    features = PolynomialFeatures(best_degree, include_bias=False)
    X_features = features.fit_transform(X)

    # fit full model with confounder C observed
    XC = np.hstack([X_features, C.reshape(-1, 1)])
    model = LinearRegression().fit(XC, y)

    # extract epsilon coefficient of confounding noise in Y
    epsilon = model.coef_.flatten()[-1]
    y_deconfounded = y - epsilon * C

    # fit f(X) = y - epsilon * C
    deconfounded = LinearRegression().fit(X_features, y_deconfounded)
    f = deconfounded.coef_.reshape(-1, 1)
    b = float(np.asarray(deconfounded.intercept_).ravel()[0])
    return f, b, features, epsilon
