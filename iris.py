import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib import pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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

#view species distribution
print(df.groupby('species').size())
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

#make a violin plot of petal length based off species
petalLengthspeciesViolin = plt.figure()
plt.title('Petal-Length By species')
sns.violinplot(data = df, x = 'species', y = 'petal-length')
petalLengthspeciesViolin.savefig(graph_folder_path + 'petalLengthspeciesViolin.png')

#make another one for sepal length
sepalLengthSpeciesViolin = plt.figure()
plt.title('Sepal-Length By species')
sns.violinplot(data = df, x = 'species', y = 'sepal-length')
sepalLengthSpeciesViolin.savefig(graph_folder_path + 'sepalLengthViolin.png')

#plot scatter plot matrix
scatterMatrix = plt.figure()
plt.title('Scatter Plot Matrix')
sns.pairplot(df, hue = 'species')
scatterMatrix.savefig(graph_folder_path + 'scatterPlot.png')

#split the dataset into a training section and testing section
train, test = train_test_split(df, test_size = 0.3)

#take data features and output for training and testing
train_x = train[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
train_y = train['species']

test_x = test[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
test_y = test['species']

#double check everything came out correctly
print(train.head)
print(' ')
print(test.head)
print(' ')

#finally start training models
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print(metrics.accuracy_score(prediction, test_y))
print(' ')

#now test the same algorithm using petal and sepal data seperately
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

#train algorithm with new data
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
