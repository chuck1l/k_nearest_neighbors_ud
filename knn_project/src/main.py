import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    df = pd.read_csv('../data/knn_project_data')
    print(df.head())
    toggle = False
    if toggle:
        sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
        plt.savefig('../imgs/pairplot_knn_data.png')
        plt.show();

    scaler = StandardScaler()
    scaler.fit(df.drop('TARGET CLASS',axis=1))
    scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
    df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
    df_feat.head()

    X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                        test_size=0.30, random_state=101)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train, y_train)
    y_pred1 = knn1.predict(X_test)
    print('Model Results 1 Neighbor: \n', classification_report(y_test,y_pred1))

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
    plt.savefig('../imgs/error_rate_knn.png')
    plt.show();
    # I like 31 Neighbors
    knn31 = KNeighborsClassifier(n_neighbors=31)
    knn31.fit(X_train, y_train)
    y_pred31 = knn31.predict(X_test)
    print('Model Results 31 Neighbors: \n', classification_report(y_test,y_pred31))
