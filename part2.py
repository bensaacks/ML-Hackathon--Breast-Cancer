from sklearn.metrics import mean_squared_error
from preprocess import *

def find_alpha_for_lasso(X,y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    real_train_x, dev_x, real_train_y, dev_y = train_test_split(train_x,
                                                    train_y, test_size=0.2)
    alphas = np.linspace(0.001, 2)
    mse_array = []
    for index, alpha in enumerate(alphas):
        lasso_reg_model = linear_model.Lasso(alpha=alpha)
        lasso_reg_model.fit(real_train_x, real_train_y)
        pred_y = lasso_reg_model.predict(dev_x)
        mse = mean_squared_error(y_true=dev_y,
                                 y_pred=pred_y)
        mse_array.append(mse)
    return np.argmin(np.array(mse_array))


def load_y2_data(X):
    y = pd.read_csv("train.labels.1.csv")
    test_x = pd.read_csv("test.feats.csv")
    test_x = preprocess_data(test_x)
    missing_cols = set(test_x.columns) - set(X.columns)
    inter_cols = set(test_x.columns) - missing_cols
    X = X[list(inter_cols)]
    test_x = test_x[list(inter_cols)]
    best_alpha = find_alpha_for_lasso(X,y)
    lasso_reg_model = linear_model.Lasso(alpha=best_alpha)
    lasso_reg_model.fit(X, y)
    pred_y = lasso_reg_model.predict(test_x)
    pd.DataFrame(pred_y).to_csv("prediction_part2", encoding='utf-8', index=False)



if __name__ == '__main__':
    np.random.seed(0)
    X = load_x_data()
    load_y2_data(X)