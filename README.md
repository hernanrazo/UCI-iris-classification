UCI Iris Classification
===

Description
---
A python script that predicts plant species based on sepal and petal lengths. The species used in this dataset are iris-setosa, iris-versicolor, iris-virginica. This example is part of the University of California - Irvine Machine Learning Repository.  

Libraries used in this example include pandas, seaborn, matplotlib, and scikit-learn. The algorithm used is the k-nearest neighbors algorithm.  

Analysis
---
First, we make box and whisker plots to see the range of values for petal and sepal dimensions.  

![petalLengthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalLengthBoxWhisker.png)  

![petalWidthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalWidthBoxWhisker.png)  

![sepalLengthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalLengthBoxWhisker.png)  


![sepalWidthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalWidthBoxWhisker.png)  

Next, plot histograms of the same data.  

![petalLengthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalLengthHist.png)  

![petalWidthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalWidthHist.png)  

![sepalLengthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalLengthHist.png)  

![sepalWidthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalWidthHist.png)  

These plots give us a good visual for the data. Now use a violin plot to condense it all into two graphs. One violin plot will show petal length and another will show sepal length.

![petalLengthViolin](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalLengthspeciesViolin.png)  

![sepalLengthViolin](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalLengthViolin.png)  

Now, since we were only given one dataset, we have to split it into a training section and testing section. Most of the data will be in the training dataset.

```python
train, test = train_test_split(df, test_size = 0.3)

#take data features and output for training and testing
train_x = train[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
train_y = train['species']

test_x = train[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
test_y = train['species']
```
This example uses the K-nearest Neighbors algorithm so use the following script to train and fit the model:  

```python
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print(metrics.accuracy_score(prediction, test_y))
print(' ')

```
This returns pretty good results but what would happen if we seperated petal and sepal lengths? To do this, again split the data into a training section and a testing section. The only difference this time is that you will to do it for both petal and sepal lengths.  

```python
#split the dataset
petal = df[['petal-length', 'petal-width', 'species']]
sepal = df[['sepal-length', 'sepal-width', 'species']]

#split the data into a training and testing section again

#petals
train_petal, test_petal = train_test_split(petal, test_size = 0.3, random_state = 0)
train_petal_x = train_petal[['petal-length', 'petal-width']]
train_petal_y = train_petal['species']

test_petal_x = test_petal[['petal-length', 'petal-width']]
test_petal_y = test_petal['species']

#sepals
train_sepal, test_sepal = train_test_split(sepal, test_size = 0.3, random_state = 0)
train_sepal_x = train_sepal[['sepal-length', 'sepal-width']]
train_sepal_y = train_sepal['species']

test_sepal_x = test_sepal[['sepal-length', 'sepal-width']]
test_sepal_y = test_sepal['species']

```  
Retrain the model for this new scenario:  

```python
print('New training session:')
#petals
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_petal_x, train_petal_y)
prediction = model.predict(test_petal_x)
print('Petal prediction: ')
print(metrics.accuracy_score(prediction, test_petal_y))
print(' ')

#sepals
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_sepal_x, train_sepal_y)
prediction = model.predict(test_sepal_x)
print('Sepal prediction: ')
print(metrics.accuracy_score(prediction, test_sepal_y))

```

It can be seen that restricting only to petal length gives a better prediction than sepal length or both. 

Acknowledgements
---
This project was made with guidance from various Kaggle kernels and other tutorials. These include [this tutorial on machinelearningmastery.com](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/) and [this IPython Notebook by I,Coder](https://www.kaggle.com/ash316/ml-from-scratch-with-iris).

Sources and Helpful Links
---
https://archive.ics.uci.edu/ml/datasets/iris  
https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python  
https://www.kaggle.com/uciml/iris/home  
https://www.kaggle.com/ash316/ml-from-scratch-with-iris