import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib import pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

#get dataset
df = pd.read_csv('/Users/hernanrazo/pythonProjects/UCI-iris-classification/data.csv')

#make a string that holds most of the folder path
graph_folder_path = '/Users/hernanrazo/pythonProjects/UCI-iris-classification/graphs/'

#double check for missing values
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')

print(df.describe())
print(' ')

#view class distribution
print(df.groupby('class').size())
print(' ')

#start visualizing and analyzing the data
#make box and whisker plots of all sepal and petal dimensions
sepalLengthBoxWhisker = plt.figure()
plt.title('Sepal-Length')
df.boxplot(column = 'sepal-length')
sepalLengthBoxWhisker.savefig(graph_folder_path + 'sepalLengthBoxWhisker.png')

sepalWidthBoxWhisker = plt.figure()
plt.title('Sepal-Width')
df.boxplot(column = 'sepal-width')
sepalWidthBoxWhisker.savefig(graph_folder_path + 'sepalWidthBoxWhisker.png')

petalLengthBoxWhisker = plt.figure()
plt.title('Petal Length')
df.boxplot(column = 'petal-length')
petalLengthBoxWhisker.savefig(graph_folder_path + 'petalLengthBoxWhisker.png')

petalWidthBoxWhisker = plt.figure()
plt.title('Petal-Width')
df.boxplot(column = 'petal-width')
petalWidthBoxWhisker.savefig(graph_folder_path + 'petalWidthBoxWhisker.png')

#make histograms for the same data
sepalLengthHist = plt.figure()
plt.title('Sepal-Length')
df['sepal-length'].hist(bins = 15)
sepalLengthHist.savefig(graph_folder_path + 'sepalLengthHist.png')

sepalWidthHist = plt.figure()
plt.title('Sepal-Width')
df['sepal-width'].hist(bins = 15)
sepalWidthHist.savefig(graph_folder_path + 'sepalWidthHist.png')

petalLengthHist = plt.figure()
plt.title('Petal-Length')
df['petal-length'].hist(bins = 15)
petalLengthHist.savefig(graph_folder_path + 'petalLengthHist.png')

petalWidthHist = plt.figure()
plt.title('Petal-Width')
df['petal-width'].hist(bins = 15)
petalWidthHist.savefig(graph_folder_path + 'petalWidthHist.png')

#make a violin plot of petal length based off class
petalLengthClassViolin = plt.figure()
plt.title('Petal-Length By Class')
sns.violinplot(data = df, x = 'class', y = 'petal-length')
petalLengthClassViolin.savefig(graph_folder_path + 'petalLengthClassViolin.png')

#plot scatter plot matrix
scatterMatrix = plt.figure()
plt.title('Scatter Plot Matrix')
sns.pairplot(df, hue = 'class')
scatterMatrix.savefig(graph_folder_path + 'scatterPlot.png')















