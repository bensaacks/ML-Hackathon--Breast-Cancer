import math
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn import linear_model
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from evaluate_part_0 import *

her2_title = 'Her2'
her2_labels = [0, 1, 2, 3]
labels1 = ['ADR - Adrenals', 'LYM - Lymph nodes', 'BON - Bones', 'BRA - Brain',
          'HEP - Hepatic', 'PUL - Pulmonary', 'PER - Peritoneum',
          'MAR - Bone Marrow', 'PLE - Pleura', 'SKI - Skin', 'OTH - Other']


def y_labels(df):
    pred_labels = parse_df_labels(df)
    enc = Encode_Multi_Hot()
    pred_vals = pred_labels["vals"]
    enc.fit(pred_vals)
    pred_multi_hot = [enc.enc(val) for val in pred_vals]
    return pred_multi_hot


def squeeze_y(y_matrix):
    y_title = "אבחנה-Location of distal metastases"
    y_vec = []
    for row in y_matrix:
        new_row = []
        for i in range(len(labels1)):
            if row[i] == 1:
                new_row.append(labels1[i])
        y_vec.append(str(new_row))
    new_df = pd.DataFrame(y_vec, columns=[y_title])
    return new_df


def preprocess_her2(df):
    df[her2_title] = df['Her2'].fillna(0)
    df.replace({her2_title: {'neg': 1, 'NEG': 1, 'Neg': 1, 'negative': 1,
                             'Neg ( FISH non amplified)': 1, 'NEGATIVE PER FISH': 1,
                             'negative by FISH': 1, 'NEGATIVE': 1, 'Negative': 1,
                             'Neg by IHC and FISH': 1, 'Neg by FISH': 1,
                             'Negative ( FISH 1.0)': 1, '0': 1, 'neg.': 1}}, inplace=True)
    df.replace({her2_title: {'-': 0, '(-)': 0}}, inplace=True)
    df.replace({her2_title: {'+2 IHC': 2, '2+': 2,
                             'Neg vs +2': 2, '+2 Fish NEG': 2, '+2 FISH-neg': 2,
                             '+2 FISH negative': 2}}, inplace=True)
    df.replace({her2_title: {'FISH pos': 3, 'Positive by FISH': 3, 'pos': 3,
                             '+3 100%cells': 3, '+3 100%': 3, 'Pos by FISH': 3, 'positive': 3,
                             'FISH POS': 3, '+2 FISH-pos': 3, '+2 FISH(-)': 3,
                             '+2, FISH חיובי': 3, 'Pos. FISH=2.9': 3, '+3 (100%cells)': 3,
                             '+2 FISH positive': 3, 'חיובי': 3}}, inplace=True)
    df[her2_title] = df[her2_title].apply(lambda x: x if x in her2_labels else 0)


def preprocess_dummies(df):
    new_df = df[["Age", "Her2", 'Tumor mark', 'Basic stage', 'Side']]

    new_df["Age"] = np.where(new_df["Age"].to_numpy() < 40, 0, 1)
    new_df = pd.get_dummies(new_df, prefix='Tumor mark', columns=['Tumor mark'])
    new_df = pd.get_dummies(new_df, prefix='Side', columns=['Side'])
    new_df = pd.get_dummies(new_df, prefix='Basic stage', columns=['Basic stage'])
    return new_df


def preprocess_with_knn(df):
    scaler = MinMaxScaler()
    new_df = pd.DataFrame(scaler.fit_transform(df),
                          columns=df.columns)
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df),
                          columns=new_df.columns)
    return df


def preprocess_data(df):
    df = df.drop(["אבחנה-Tumor depth", "אבחנה-Surgery name3",
                  "אבחנה-Surgery name2", "אבחנה-Tumor width",
                  "אבחנה-Surgery date3", "אבחנה-Surgery date2",
                  "אבחנה-Ivi -Lymphovascular invasion"], axis=1)
    df.rename(columns={'אבחנה-Her2': 'Her2', 'אבחנה-Age': 'Age',
                       'אבחנה-T -Tumor mark (TNM)': 'Tumor mark',
                    'אבחנה-Basic stage': 'Basic stage', 'אבחנה-Side': 'Side',
                       'אבחנה-Nodes exam': "Nodes_Exam"}, inplace=True)

    df = df[["Age", "Her2", 'Tumor mark', 'Basic stage', 'Side', 'Nodes_Exam']]
    df = preprocess_dummies(df)
    preprocess_her2(df)
    df = preprocess_with_knn(df)

    return df


def load_x_data():
    X = pd.read_csv("train.feats.csv", dtype={'אבחנה-Surgery name3': 'str',
                                "אבחנה-Ivi -Lymphovascular invasion": 'str',
                                              'אבחנה-Surgery date3': 'str'})
    return preprocess_data(X)