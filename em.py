from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\\ml\\IRIS.csv")
X = dataset.iloc[:, :-1]
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = dataset.iloc[:, -1].map(label)
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.petal_length, X.petal_width, c=colormap[y])
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.petal_length, X.petal_width, c=colormap[y_cluster_gmm])  
print('The accuracy score of GMM:', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of GMM:\n', metrics.confusion_matrix(y, y_cluster_gmm))
plt.tight_layout()
plt.show()
