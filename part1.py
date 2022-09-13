from preprocess import  *

def load_y1_data(X):
    y = pd.read_csv("train.labels.0.csv")
    test_x = pd.read_csv("test.feats.csv")
    test_x = preprocess_data(test_x)
    missing_cols = set(test_x.columns) - set(X.columns)
    inter_cols = set(test_x.columns) - missing_cols
    X = X[list(inter_cols)]
    test_x = test_x[list(inter_cols)]
    y_train_processed = y_labels(y)

    # DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y_train_processed)
    y_pred = list(decision_tree.predict(test_x))
    y_vec = squeeze_y(y_pred)
    y_vec.to_csv("prediction_part1", encoding='utf-8', index=False)

if __name__ == '__main__':
    np.random.seed(0)
    X = load_x_data()

    load_y1_data(X)
