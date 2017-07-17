import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics


# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


if __name__ == '__main__':
    original_data = pd.read_csv(
        "adult.data.txt",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

    print("Data with missing values set to Nan:")
    print(original_data)
    print('\n')

    encoded_data, _ = number_encode_features(original_data)
    print( "Encoded Data:")
    print( encoded_data.head())
    print( '\n')

    # Split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_data[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country"]], encoded_data["Target"], train_size=0.70)

    # Scale the data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
    X_test = scaler.transform(X_test.astype("float64"))

    print( "X_train:")
    print( X_train.head())
    print( '\n')

    print( "Y_train:")
    print( y_train.head())
    print( '\n')

    print( "X_test:")
    print(X_test)
    print('\n')

    print("Y_test:")
    print(y_test.head())
    print( '\n')

    # Define model
    cls = linear_model.LogisticRegression()

    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    print( cm)
    print( "F1 score: %f" % sklearn.metrics.f1_score(y_test, y_pred))