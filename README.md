UCI Iris Classification
===

Description
---
A python script that predicts plant species based on sepal and petal lengths. The species used in this dataset are iris-setosa, iris-versicolor, iris-virginica. This example is part of the University of California - Irvine Machine Learning Repository.  

Libraries used in this example include pandas, seaborn, matplotlib, and scikit-learn. The algorithm used is the k-nearest neighbors algorithm.  

Analysis
---
First, we make bo and whisker plots to see the range of values for petal and sepal dimensions.  

![petalLengthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalLengthBoxWhisker.png)  

![petalWidthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalWidthBoxWhisker.png)  

![sepalLengthBW](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalLengthBoxWhisker.png)  

![sepalWidthBWcom/hrazo7/UCI-iris-classification/blob/master/graphs/sepalWidthBoxWhisker.png)  

Next, plot histograms of the same data.  

![petalLengthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalLengthHist.png)  

![petalWidthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/petalWidthHist.png)  

![sepalLengthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalLengthHist.png)  

![sepalWidthHist](https://github.com/hrazo7/UCI-iris-classification/blob/master/graphs/sepalWidthHist.png)  

These plots give us a good visual for the data. Now use a violin plot to condense it all to two graphs. One violin plot will show petal length and another will show sepal length.

! 



Sources and Helpful Links
---
https://archive.ics.uci.edu/ml/datasets/iris  
https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python  
https://www.kaggle.com/uciml/iris/home  
https://www.kaggle.com/ash316/ml-from-scratch-with-iris