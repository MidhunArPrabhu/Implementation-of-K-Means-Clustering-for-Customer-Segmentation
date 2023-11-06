
# EXPERIMENT-08

# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.choose the number of clusters

Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2. Initialize cluster centroids
   
Randomly select K data points from your dataset as the initial centroids of the clusters.

4. Assign data points to clusters
   
Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

6. Update cluster centroids
   
Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

8. Repeat steps 3 and 4
   
Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

10. Evaluate the clustering results
    
Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

12. Select the best clustering solution
    
If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```py
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber: 212222240066

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
### data.head():

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/cf04c06f-571f-48a2-ab55-273a3e15457f)

### data.info():

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/a143839a-1d16-48d4-ba98-5ca631bedf4b)

### NULL VALUES:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/73d722d4-d939-4152-acf1-e6f886ccb1e4)

### ELBOW GRAPH:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/cbf7328c-56d6-49ef-bfc5-81d989504ed9)

### CLUSTER FORMATION:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/480f6100-e1f6-443d-8c2b-faed554647b0)

### PREDICTED VALUE:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/ed677eec-e15b-444e-afc1-773aaa45373d)

### FINAL GRAPH(D/O):

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393818/98f6ffd9-c24c-440c-a33e-87740b667b98)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
