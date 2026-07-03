from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

def train_healthy_classifier(X, y):
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def train_overfitted_classifier(X, y):
    # Very deep tree that will memorize the training data
    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    model.fit(X, y)
    return model

def train_healthy_regression(X, y):
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def train_overfitted_regression(X, y):
    model = DecisionTreeRegressor(max_depth=None, random_state=42)
    model.fit(X, y)
    return model
