# TitanicSurvivalPrediction
This project uses a dataset which consists of different information about the passengers onboard the Titanic to predict whether those passengers survived or not. This dataset is from Kaggle.
The dataset consists of training set and testing set.The training set contains data we can use to train our model. It has a number of feature columns which contain various descriptive data, as well as a column of the target values we are trying to predict: in this case, Survival.The testing set contains all of the same feature columns, but is missing the target value column.
we use pandas.read_csv() library to read both our files train.csv and test.csv and inspect their size.
The features present in the dataset are following:
      survival: Survival(1 means yes,0 means no)
      PassengerId: Unique Id of a passenger
      pclass: Ticket class
      sex: Sex
      Age: Age in years
      sibsp: # of siblings / spouses aboard the Titanic
      parch: # of parents / children aboard the Titanic
      ticket: Ticket number
      fare: Passenger fare
      cabin: Cabin number
      embarked: Port of Embarkation
      
The features consist of numerical as well as categorical values.The dataset also has some missing values(NaN).
It is clear from the dataset that some variables will be responsible in predicting the output while some will have no effect(like passengerId,ticket number etc).
 Age, Sex, and PClass may be good predictors of survival. We’ll start by exploring Sex and Pclass by visualizing the data.
 We use bar plot to determine the role of gender of the passengers,and we see that females survived in much higher proportions than males.
 Similarly, we see the effect passenger class has on the output.The passengers that survived from first class are in much higher proportions than the ones in third class,while the passengers from second class lie in between.
After further study of the features we have to prepare our dataset.As discussed earlier our dataset consists of missing values as well as categorical values.In order to build our model we need to fill these missing values and convert the categorical values into numerical values.

We change the Sex and Embarked variables to numerical data
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})
train['Embarked']=train['Embarked'].map({'S':1,'C':2,'Q':3})
test['Embarked']=test['Embarked'].map({'S':1,'C':2,'Q':3})

And we impute the missing values in the Age and Embarked column
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mean(),inplace=True)
test['Embarked'].fillna(test['Embarked'].mean(),inplace=True)

SPLITTING OUR DATA
we do have a test dataframe that we could use to make predictions. We could make predictions on that data set, but because it doesn’t have the Survived column we would have to submit it to Kaggle to find out our accuracy. This would be difficult as we would have to submit to find out the accuracy every time we optimized our model.
We could also fit and predict on our train dataframe, however if we do this there is a high likelihood that our model will overfit, which means it will perform well because we’re testing on the same data we’ve trained on, but then perform much worse on new, unseen data.

Instead we can split our train dataframe into two:
One part to train our model on (often 80% or 70% of the observations)
One part to make predictions with and test our model (often 20% or 30% of the observations)

We use the scikit-learn library model_selection.train_test_split() function that we can use to split our data. train_test_split() accepts two parameters, X and y, which contain all the data we want to train and test on, and returns four objects: train_X, train_y, test_X, test_y:

The type of machine learning we will be doing is called classification, because when we make predictions we are classifying each passenger as ‘survived’ or not.
We use Logistic Regression for classification.
We will be using the scikit-learn library as it has many tools that make performing machine learning easier. The scikit-learn workflow consists of four main steps:
Instantiate (or create) the specific machine learning model you want to use
Fit the model to the training data
Use the model to make predictions
Evaluate the accuracy of the predictions

We use the LogisticRegression.fit() method to train our model. The .fit() method accepts two arguments: X and y. X must be a two dimensional array (like a dataframe) of the features that we wish to train our model on, and y must be a one-dimensional array (like a series) of our target, or the column we wish to predict.

Once we have fit our model, we can use the LogisticRegression.predict() method to make predictions.
The predict() method takes a single parameter X, a two dimensional array of features for the observations we wish to predict. X must have the exact same features as the array we used to fit our model. The method returns single dimensional array of predictions.

scikit-learn has a handy function we can use to calculate accuracy: metrics.accuracy_score(). The function accepts two parameters,the actual values and our predicted values respectively, and returns our accuracy score.

Our model has an accuracy score of 79.88% when tested against our 20% test set. Given that this data set is quite small, there is a good chance that our model is overfitting, and will not perform as well on totally unseen data.

To give us a better understanding of the real performance of our model, we can use a technique called cross validation to train and test our model on different splits of our data, and then average the accuracy scores.
We’ll use model_selection.cross_val_score() to perform cross-validation on our data, before calculating the mean of the
scores produced
