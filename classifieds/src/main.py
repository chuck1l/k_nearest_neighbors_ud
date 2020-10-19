import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    df = pd.read_csv('../data/classified_data.csv', index_col=0)
    print(df.head())
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('TARGET CLASS', axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    print(df_feat.head())
    X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                        test_size=0.30, random_state=101)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train, y_train)
    y_pred1 = knn1.predict(X_test)
    print('Results With 1 Neighbor: \n', classification_report(y_test, y_pred1))
    # Identify best K value
    error_rate = []
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.savefig('../imgs/error_rate_plot.png');
    # I like 12, 18, 34 neighbors
    knn12 = KNeighborsClassifier(n_neighbors=12)
    knn12.fit(X_train, y_train)
    y_pred12 = knn12.predict(X_test)
    print('Results With 12 Neighbor: \n', classification_report(y_test, y_pred12))
    # 18 neighbors
    knn18 = KNeighborsClassifier(n_neighbors=18)
    knn18.fit(X_train, y_train)
    y_pred18 = knn18.predict(X_test)
    print('Results With 18 Neighbor: \n', classification_report(y_test, y_pred18))
    # 34 neighbors
    knn34 = KNeighborsClassifier(n_neighbors=34)
    knn34.fit(X_train, y_train)
    y_pred34 = knn34.predict(X_test)
    print('Results With 34 Neighbor: \n', classification_report(y_test, y_pred34))




