{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NumPy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**SAAD SHAIKH, 20070328**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing the required Libraries\n",
    "\n",
    "import numpy as np\n",
    "from collections import Counter\n",
    "import operator\n",
    "import matplotlib.pyplot as plt\n",
    "from math import *\n",
    "import pandas as pd\n",
    "from sklearn.metrics import accuracy_score\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "# available sampe data files\n",
    "# classification: class1.csv, class2.csv\n",
    "# regression: regr1.csv, regr2.csv\n",
    "\n",
    "class_file_name = \"class1.csv\" #classification filename\n",
    "regr_file_name  = \"regr1.csv\" #regression filename"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The sample data is structured in the following structure:\n",
    "$$\n",
    "\\begin{array}{cccc}\n",
    "\\vec{y} & \\vec{x}_1 & \\vec{x}_2 & \\cdots \\\\ \\hline \n",
    "y_1 & x_{11} & x_{12} & \\cdots \\\\\n",
    "y_2 & x_{21} & x_{22} & \\cdots \\\\\n",
    "y_3 & x_{31} & x_{32} & \\cdots \\\\\n",
    "\\vdots & \\vdots & \\vdots & \n",
    "\\end{array}\n",
    "$$\n",
    "\n",
    "where $\\vec{y}$ is a column-vector of responses and $\\vec{x}_1$, $\\vec{x}_2$, $\\ldots$ are column-vectors of predictors. For the classification problem $y_i$'s take integer values: 1, 2, 3, etc. For the regression problem $y_i$'s are real numbers.\n",
    "\n",
    "**Important!** Your code must work well for any data file having this structure."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1 - KNN classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.1 KNN classification algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section you should write a function ``knn_classify(test, train, k)`` that takes train and test data as numpy ndarrays, and a k-value as an integer, and returns the class-values of the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_file = pd.read_csv('class1.csv')\n",
    "cols = ['x1', 'x2', 'y']\n",
    "df = class_file[cols].sample(frac=1).values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "reg_file_name = pd.read_csv('regr1.csv')\n",
    "cols = ['x1', 'x2', 'y']\n",
    "df_reg = reg_file_name[cols].sample(frac=1).values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of the KNN Classifier is :  66.25\n"
     ]
    }
   ],
   "source": [
    "# Defining Function for calculating Euclidean Distance\n",
    "\n",
    "def euclidean_distance(x1,x2):\n",
    "    distance = 0.0\n",
    "    for i in range(len(x1)-1):\n",
    "        distance += (x1[i]-x2[i])**2\n",
    "    return sqrt(distance)\n",
    "\n",
    "# Defining function to split the data into train, validation and test sets.\n",
    "\n",
    "def data_split(X,train_size=0.6,test_size=0.2):\n",
    "    val_size = 1-train_size-test_size\n",
    "    train,val,test = X[:int(len(X)*train_size)],X[int(len(X)*train_size):int((len(X)*(train_size+val_size)))],X[int((len(X)*(train_size+val_size))):]\n",
    "    return(train,val,test)\n",
    "\n",
    "#Defining function to classify using KNN algorthim\n",
    "\n",
    "def knn_classify(test,train,k):\n",
    "    p = []\n",
    "    for i in test: \n",
    "        D = []\n",
    "        for idx,j in enumerate(train):    #iteration of data between train data\n",
    "            dist = euclidean_distance(j, i)    #calculating Euclidean distance\n",
    "            D.append([dist,j]) \n",
    "        distances = sorted(D, key=lambda x: x[0]) #sorting distances\n",
    "        nb = [i[1] for i in D[:k]]\n",
    "        labels = sorted(dict(Counter([i[-1] for i in nb])).items(),key=operator.itemgetter(1), reverse=True)[0][0]\n",
    "        p.append(labels) #.append for the collection of lables\n",
    "    return p\n",
    "    \n",
    "\n",
    "train,validation,test = data_split(X=df,train_size=0.6,test_size=0.4) #split the data as per train-60% and test-40%\n",
    "y_pred = knn_classify(test,train,k=2) #calculating the predicted values\n",
    "y_test = [i[-1] for i in test]\n",
    "\n",
    "accuracy = accuracy_score(y_pred,y_test)\n",
    "print(\"Accuracy of the KNN Classifier is : \",accuracy*100) #calculating the accuracy of predicted and actual"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2 Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section you should read the data. Then split it randomly into train (60%), validation (20%), and test (20%) data. Use the train and validation data to find k-value giving the best classification result. Then use this k-value to classify the test data and report your findings: the k-value and the percentage of correct predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Best K value is: 19\n",
      "The accuracy of the tesi is 62.5\n"
     ]
    }
   ],
   "source": [
    "# Splitting the data into train-60%, val-20% and test-20%\n",
    "\n",
    "train,val,test = data_split(X=df,train_size=0.6,test_size=0.2)\n",
    "\n",
    "accuracy_list = []\n",
    "for i in range(1,20):  #iterating between 1 - 20 of k values\n",
    "    y_pred = knn_classify(val,train,k=i) #Calling Knn_classify to predict the target variable\n",
    "    y_valid = [i[-1] for i in val]\n",
    "    accuracy  = accuracy_score(y_pred,y_valid)  #calculating the accuarcy\n",
    "    accuracy_list.append(accuracy)  # appending all the accuracies\n",
    "best_k_value = np.argsort(accuracy_list)[-1]+1  #sorting out from the accuarcies\n",
    "print(\"The Best K value is:\",best_k_value) #printing the values of Best k value\n",
    "\n",
    "# calculating the test accuracy with the help of best_K_value obtained above\n",
    "\n",
    "y_test_pred = knn_classify(test,train,k=best_k_value)\n",
    "y_test = [i[-1] for i in test]\n",
    "accuracy2 = accuracy_score(y_test_pred,y_test)\n",
    "print('The accuracy of the tesi is',accuracy2*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2 - KNN and linear regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 KNN regression algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section you should write a function ``knn_regression(train, test, k)`` that takes train and test data, and a k-value, and returns the regression (fitted) values of the responses of the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Residual sum of squares(RSS SCORE) using KNN Regressor :  0.075\n"
     ]
    }
   ],
   "source": [
    "#Defining the function to calculate Residual sum of squares\n",
    "\n",
    "def RSS(y_pred,y_test):\n",
    "    return(np.mean([(j-i)**2 for i,j in zip(y_pred,y_test)]))\n",
    "\n",
    "#Defining the Function for KNN regression algorthim\n",
    "def knn_regression(test,train,k):\n",
    "    y_pred2 = []\n",
    "    for i in test: #iterating between the test data\n",
    "        distances2 = []\n",
    "        for idx,j in enumerate(train): #iterating between train point\n",
    "            dist = euclidean_distance(j, i) #calculating euclidean distance\n",
    "            distances2.append([dist,j])\n",
    "        distances2 = sorted(distances2, key=lambda x: x[0]) # sorting out by  distances\n",
    "        neighbors = [i[1] for i in distances2[:k]] # as per given K value choosing neighbors\n",
    "        lab = sorted(dict(Counter([i[-1] for i in neighbors])).items(),key=operator.itemgetter(1), reverse=True)[0][0]\n",
    "        y_pred2.append(lab)\n",
    "    return y_pred2\n",
    "\n",
    "# Splitting the Data with Train 60% and Test 40%\n",
    "train2,val2,test2 = data_split(X=df_reg,train_size=0.6,test_size=0.4)\n",
    "\n",
    "y_pred2 = knn_regression(test,train,k=2) #taking a random K value\n",
    "y_test2 = [i[-1] for i in test]\n",
    "print(\"Residual sum of squares(RSS SCORE) using KNN Regressor : \", RSS(y_pred2,y_test)) #calculating RSS Score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Linear regression algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section you should write a function ``linear_regression(train, test)`` that takes train and test data, and returns linear regression (fitted) values of the responses of the test data. The column-vector of regression values $\\hat{\\vec y}$ should be computed using this formula:\n",
    "\n",
    "$$\n",
    "\\hat{\\vec y} = X^{(test)} \\hat{\\vec \\beta} \n",
    "$$\n",
    "\n",
    "where \n",
    "\n",
    "- $X^{(test)}$ is the test design matrix obtained by stacking together a column of 1's with columns of predictors variables from the test data:\n",
    "\n",
    "$$\n",
    "X^{(test)} = \\begin{bmatrix} 1 & \\vec x^{(test)}_1 & \\vec x^{(test)}_2 & \\cdots \\end{bmatrix} = \\begin{bmatrix} \n",
    "1 & x^{(test)}_{11} & x^{(test)}_{12} & \\cdots \\\\ \n",
    "1 & x^{(test)}_{21} & x^{(test)}_{22} & \\cdots \\\\\n",
    "\\vdots & \\vdots & \\vdots \\\\\n",
    "1 & x^{(test)}_{m1} & x^{(test)}_{m2} & \\cdots\\end{bmatrix}\n",
    "$$\n",
    "\n",
    "- $\\hat{\\vec \\beta}$ is a column vector of least-squares estimates of the regression coefficients:\n",
    "\n",
    "$$\n",
    "\\hat{\\vec \\beta} = \\big((X^{(train)})^T X^{(train)} \\big)^{-1} (X^{(train)})^T \\vec y^{(train)}\n",
    "$$\n",
    " \n",
    "- $X^{(train)}$ is the design matrix for the train data:\n",
    "\n",
    "$$\n",
    "X^{(train)} = \\begin{bmatrix} 1 & \\vec x^{(train)}_1 & \\vec x^{(train)}_2 & \\cdots \\end{bmatrix} = \\begin{bmatrix} \n",
    "1 & x^{(train)}_{11} & x^{(train)}_{12} & \\cdots \\\\ \n",
    "1 & x^{(train)}_{21} & x^{(train)}_{22} & \\cdots \\\\\n",
    "\\vdots & \\vdots & \\vdots \\\\\n",
    "1 & x^{(train)}_{n1} & x^{(train)}_{n2} & \\cdots\\end{bmatrix}\n",
    "$$\n",
    "\n",
    "- $m$ is the number of rows of the test data\n",
    "\n",
    "- $n$ is the number of rows of the train data\n",
    "\n",
    "- $\\vec y^{(train)}$ is a column-vector of response values of the train data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RSS (Residual sum of scores) SCORE :  0.6377750243204184\n"
     ]
    }
   ],
   "source": [
    "#Defining the function for cost function\n",
    "\n",
    "def cost_f(X, y, P):\n",
    "    return (np.sum(X.dot(P)-y)**2)/(2*(len(y)))\n",
    "\n",
    "# Defining Gradient Function for better coefficients\n",
    "\n",
    "def batch_gradient(X,y,P,learning_rate,iterations):\n",
    "    cost_hist = list()\n",
    "    for iteration in range(iterations):\n",
    "        loss_values = (X.dot(P)-y) # loss clculation\n",
    "        gradients = ((X.T.dot(loss_values)) / (len(y))) #calculation of the gradients\n",
    "        P = P - (learning_rate* gradients) #modification of the existing coefficients\n",
    "        cost_value = cost_f(X, y, P) # calling out cost function for caculating the cost as per X,y and modfied P coefficients\n",
    "        cost_hist.append(cost_value) # appending the cost value\n",
    "    return P,cost_hist  # returning the Best coefficents value and Cost value\n",
    "\n",
    "\n",
    "#Defining the function for  Linear regression \n",
    "\n",
    "def Linear_Reg(train, test,iterations=2000, learning_rate =0.005):\n",
    "    X_train, y_train = train[:,:-1],train[:,-1] #  train data\n",
    "    X_test, y_test = test[:,:-1],test[:,-1] #test data\n",
    "    P = np.zeros(X_train.shape[1]) # generating random coefficients\n",
    "    modif_p,cost = batch_gradient(X_train, y_train, P,learning_rate=learning_rate,iterations=iterations)\n",
    "    y_pred = X_test.dot(modif_p)  # predicting the test/validation data\n",
    "    return y_pred,modif_p\n",
    "\n",
    "#calculations of the predictions with  train-60% and test-40% data \n",
    "\n",
    "y_pred,best_p = Linear_Reg(train, test)\n",
    "y_test = [i[-1] for i in test]\n",
    "\n",
    "#Metrics choosen is Residual sum of squares\n",
    "print(\"RSS (Residual sum of scores) SCORE : \",RSS(y_pred,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section you should read the data. Then split it randomly into train (60%), validation (20%), and test (20%) data. Use the train and validation data to find k-value giving the best knn regression result. Then use this k-value to conduct knn regression on the test data and report your findings: the k-value and the [residual sum of squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares): $RSS = \\sum_{i=1}^m (\\hat{y}_i - y_i)^2$ where $\\hat{y}_i$ are predicted values, $y_i$ are observed values, and $m$ is the number of observations in your test data. Then repeat the last step using the linear regression approach. Finally, compare the the two RSS values your have obtained. Which algorithm, knn or linear regression, gives a better result?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RSS for K- 1 :  0.8182400000000001\n",
      "RSS for K- 2 :  0.8182400000000001\n",
      "RSS for K- 3 :  0.8182400000000001\n",
      "RSS for K- 4 :  0.8226275000000001\n",
      "RSS for K- 5 :  0.7896675\n",
      "RSS for K- 6 :  0.80395\n",
      "RSS for K- 7 :  0.8738949999999999\n",
      "RSS for K- 8 :  0.8533125\n",
      "RSS for K- 9 :  0.8127675\n",
      "RSS for K- 10 :  0.8127675\n",
      "RSS for K- 11 :  0.8009774999999999\n",
      "RSS for K- 12 :  0.8016174999999999\n",
      "RSS for K- 13 :  0.748045\n",
      "RSS for K- 14 :  0.7017325\n",
      "RSS for K- 15 :  0.6710974999999999\n",
      "RSS for K- 16 :  0.668635\n",
      "RSS for K- 17 :  0.668845\n",
      "RSS for K- 18 :  0.6639225000000001\n",
      "RSS for K- 19 :  0.6639225000000001\n",
      "--------------------\n",
      "Best K value is : 16\n"
     ]
    }
   ],
   "source": [
    "#Splitting the dataset according to train-60%,val-20% and test-20%\n",
    "\n",
    "train,val,test = data_split(X=df_reg,train_size=0.60,test_size=0.2)\n",
    "\n",
    "\n",
    "residual = []\n",
    "for i in range(1,20):\n",
    "    y_pred = knn_regression(val,train,k=i)\n",
    "    y_values = [i[-1] for i in val]\n",
    "    r  = RSS(y_pred,y_values) #calculating RSS \n",
    "    print(f\"RSS for K- {i} : \", RSS(y_pred,y_test)) \n",
    "    residual.append(r)# appending the RSS value\n",
    "best_k_value2 = np.argsort(residual)[0]+1 # taking the postion of loww RSS value\n",
    "print('-'*20)\n",
    "print(\"Best K value is :\",best_k_value2) \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Residual sum of squares(RSS Score) for KNN :  0.19098375\n"
     ]
    }
   ],
   "source": [
    " # predicting the test using KNN regression\n",
    "\n",
    "y_test_predict = knn_regression(test,train,k=best_k_value)\n",
    "y_test = [i[-1] for i in test]\n",
    "print(\"Residual sum of squares(RSS Score) for KNN : \",RSS(y_test_predict,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rate = [0.3, 0.2, 0.1, 0.01, 0.001 ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RSS SCORE IS :  0.12738081313695132\n",
      "RSS SCORE IS :  0.1273808131369511\n",
      "RSS SCORE IS :  0.12738080968898602\n",
      "RSS SCORE IS :  0.13381289330167426\n",
      "RSS SCORE IS :  0.3988348272720893\n"
     ]
    }
   ],
   "source": [
    "for lr in learning_rate:\n",
    "    y_hat,best_b = Linear_Reg(train, test, learning_rate = lr)\n",
    "    y_test = [i[-1] for i in test]\n",
    "    print(\"RSS SCORE IS : \",RSS(y_hat,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The RSS (Residual sum of squares) value for Linear Regression is almost constant across all learning rate except the last one. After seeing the output, the knn algorithm beats the linear regression algorithm."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
