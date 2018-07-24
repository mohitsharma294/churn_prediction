# importing Basic required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading train and test file
churn_data_df = pd.read_csv("Train_data.csv")
test_data_df = pd.read_csv("Test_data.csv")
############################################
#                                          #
#     2.1 Exploratory Data Analysis        #
#                                          #
############################################

###################################
#  2.1.1 understanding the data   #
###################################

# checking dimension of data
print(churn_data_df.shape)
print(test_data_df.shape)
# looking at few observation 
churn_data_df.head()
# all columns of dataset
churn_data_df.columns
# Checking datatypes and information of dataset
churn_data_df.info()
# Checking numerical statistics of continuous variable
churn_data_df.describe()

# Extracting each category with object datatype and adding area code as area code is
# category values in numerical form
cat_columns = list(churn_data_df.columns[churn_data_df.dtypes == 'object'])
cat_columns.insert(2, 'area code')
cat_columns
# changing to categorical variable to category datatype
churn_data_df[cat_columns] = churn_data_df[cat_columns].apply(pd.Categorical)
test_data_df[cat_columns] = test_data_df[cat_columns].apply(pd.Categorical)
# checking total unique values in each categorical variable
churn_data_df[cat_columns].nunique()

# counting of each unique values in last three category
churn_data_df[cat_columns[3:6]].apply(pd.Series.value_counts)

# alternate solution to getting counting in one go
print("value counts of categorical column")
print()
for i in cat_columns[2:6]:
    print(i)
    print(churn_data_df[i].value_counts())
    print("=================================")

# getting percentage of target variable Churn in training dataset
churn_data_df['Churn'].value_counts(normalize = True)

###################################
#  2.1.2 Missing value analysis   #
###################################
# checking for missing value in each columns
churn_data_df.isnull().sum()

###################################
#  2.1.3 outlier analysis         #
###################################

# defining function to plot historgram and box plot of numerical variable
def hist_and_box_plot(col1, col2, col3, data, bin1=30, bin2=30, bin3=30, sup =""):
    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize= (12,6))
    super_title = fig.suptitle("Boxplot and Histogram: "+sup, fontsize='x-large')
    plt.tight_layout()
    sns.boxplot(y = col1, x = 'Churn', data = data, ax = ax[0][0], hue = 'Churn')
    sns.boxplot(y = col2, x = 'Churn', data = data, ax = ax[0][1], hue = 'Churn')
    sns.boxplot(y = col3, x = 'Churn', data = data, ax = ax[0][2], hue = 'Churn')
    sns.distplot(data[col1], ax = ax[1][0], bins = bin1)
    sns.distplot(data[col2], ax = ax[1][1], bins = bin2)
    sns.distplot(data[col3], ax = ax[1][2], bins = bin3)
    fig.subplots_adjust(top = 0.90)
    plt.show()

    
# plotting histogram and boxplot for day calls, minute and charges
hist_and_box_plot('total day minutes', 'total day calls', 'total day charge', 
                  data = churn_data_df, sup = "Day time call details")

# plotting histogram and boxplot for evening calls, minute and charges
hist_and_box_plot('total eve minutes', 'total eve calls', 'total eve charge', 
                  data = churn_data_df, sup = "Evening call details")

# plotting histogram and boxplot for night calls, minute and charges
hist_and_box_plot('total night minutes', 'total night calls', 'total night charge', 
                  data = churn_data_df, sup = "Night call details")

# plotting histogram and boxplot for international calls, minute and charges
hist_and_box_plot('total intl minutes', 'total intl calls', 'total intl charge', 
                  data = churn_data_df, bin2=10,sup="International call Details")

# plot for account length , vmail messages and customer service calls
hist_and_box_plot('account length','number vmail messages','number customer service calls', 
                  data = churn_data_df, bin2 = 10, bin3 = 5,
                  sup = "account, vmail messages, customer service calls")

#####################
# outlier removing  #
#####################
# making another dataset which will not contain outlier stated by boxplot
# as we dont want to loose information already we have small dataset, 
# so will create two dataset
# further reasorn explained in Project report
# churn_data_df_wo will be our second dataset without outliers
churn_data_df_wo = churn_data_df
# getting all numeric columns
numeric_columns = list(churn_data_df.columns[churn_data_df.dtypes != 'category'])
# removing numeric columns for which we will not do outlier removal process
numeric_columns.remove('number vmail messages')
numeric_columns.remove('number customer service calls')

# removing outliers with boxplot method i.e. points which lie below 1.5*IQR distance
# and above 1.5*IQR distance from median
for i in numeric_columns:
     q75, q25 = np.percentile(churn_data_df_wo.loc[:,i], [75 ,25])
     iqr = q75 - q25
     min = q25 - (iqr*1.5)
     max = q75 + (iqr*1.5)
     churn_data_df_wo = churn_data_df_wo.drop(
             churn_data_df_wo[churn_data_df_wo.loc[:,i] < min].index)
     churn_data_df_wo = churn_data_df_wo.drop(
             churn_data_df_wo[churn_data_df_wo.loc[:,i] > max].index)

# plotting histogram and boxplot for day calls, minute and charges for churn_data_df_wo
hist_and_box_plot('total day minutes', 'total day calls', 'total day charge', 
                  data = churn_data_df_wo, sup = "Day time call details")

#plotting histogram and boxplot for evening calls, minute and charges for churn_data_df_wo
hist_and_box_plot('total eve minutes', 'total eve calls', 'total eve charge', 
                  data = churn_data_df_wo, sup = "Evening call details")

# plotting histogram and boxplot for night calls, minute and charges for churn_data_df_wo
hist_and_box_plot('total night minutes', 'total night calls', 'total night charge', 
                  data = churn_data_df_wo, sup = "Night call details")

#plotting histogram and boxplot for international detail for churn_data_df_wo
hist_and_box_plot('total intl minutes', 'total intl calls', 'total intl charge',
                  data = churn_data_df_wo, bin2=10, sup="International call Details")

###################################
#  2.1.4 Feature Selection        #
###################################

# Correlation plot between numerical values
numeric_columns = list(churn_data_df.columns[churn_data_df.dtypes != 'category'])
sns.pairplot(data = churn_data_df, x_vars= numeric_columns, y_vars= numeric_columns,
             hue = 'Churn')

# heat map plot between numerical values
fig = plt.figure(figsize = (14,10))
corr = churn_data_df[numeric_columns].corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), square = True,
            annot= True, cmap = sns.diverging_palette(220, 10, as_cmap= True))
plt.title("HeatMap between numerical columns of churn dataset")

# checking dependency between churn and independent variable (category)
cat_var = ['state', 'area code', 'international plan', 'voice mail plan']
from scipy.stats import chi2_contingency
print("Chi-square - test of independence")
print("=================================")
for i in cat_var:
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(churn_data_df['Churn'], 
                                                    churn_data_df[i]))
    print("p-value between Churn and {}".format(i))
    print(p)
    print('----------------------------')
    
# checking independency between independent variables
chi2, p, dof, ex = chi2_contingency(pd.crosstab(churn_data_df['international plan'],
                                                churn_data_df['voice mail plan']))
print("p-value between international plan  and voice mail plan")
print(p)
print('----------------------------')

# Dropping state, area code and phone number as they are not giving infomation 
churn_data_df = churn_data_df.drop(columns=['state', 'area code', 'phone number'])
churn_data_df_wo = churn_data_df_wo.drop(columns=['state','area code','phone number'])
test_data_df = test_data_df.drop(columns=['state', 'area code', 'phone number'])
# changing categories to levels (0 and 1)
numeric_columns = list(churn_data_df.columns[churn_data_df.dtypes != 'category'])
cat_columns = churn_data_df.columns[churn_data_df.dtypes == 'category']
for i in cat_columns:
    churn_data_df[i] = churn_data_df[i].cat.codes
    churn_data_df_wo[i] = churn_data_df_wo[i].cat.codes
    test_data_df[i] = test_data_df[i].cat.codes

# checking importance of feature
from sklearn.ensemble import ExtraTreesClassifier
cls = ExtraTreesClassifier(n_estimators=200)
X = churn_data_df.drop(columns=['Churn'])
y = churn_data_df['Churn']
cls.fit(X, y)
imp_feat = pd.DataFrame({'Feature': churn_data_df.drop(columns=["Churn"]).columns,
                         'importance':cls.feature_importances_})
imp_feat.sort_values(by = 'importance', ascending=False).reset_index(drop = True)

# Checking VIF values of numeric columns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
from statsmodels.tools.tools import add_constant
numeric_df = add_constant(churn_data_df[numeric_columns])
vif = pd.Series([vf(numeric_df.values, i) for i in range(numeric_df.shape[1])], 
                 index = numeric_df.columns)
vif.round(1)

# Deleting multicollinear columns
churn_data_df=churn_data_df.drop(columns=['total day minutes','total eve charge',
                                          'total night charge', 'total intl charge'])
churn_data_df_wo=churn_data_df_wo.drop(columns=['total day minutes','total eve charge',
                                                'total night charge',
                                                'total intl charge'])
test_data_df = test_data_df.drop(columns=['total day minutes','total eve charge',
                                          'total night charge', 'total intl charge'])

# checking again VIF after removal of multicollinear columns
numeric_columns = list(churn_data_df.columns[3:13])
numeric_columns.insert(0, 'account length')
numeric_df = add_constant(churn_data_df[numeric_columns])
vif = pd.Series([vf(numeric_df.values, i) for i in range(numeric_df.shape[1])], 
                 index = numeric_df.columns)
vif.round(1)

# splitting in X and y for train and test
# X_train -> whole datset
# X_train_wo -> dataset after removal of outliers
X_train = churn_data_df.drop('Churn', axis = 1)
y_train = churn_data_df['Churn']
X_train_wo = churn_data_df_wo.drop('Churn', axis =1)
y_train_wo = churn_data_df_wo['Churn']
X_test = test_data_df.drop('Churn', axis = 1)
y_test = test_data_df['Churn']

############################################
#                                          #
#                                          #
#   2.2.2 Building Classification models   #
#                                          #
#                                          #
############################################

# making general function to fit and predict result (Confusion Matrix) 
# and performance (K-fold CV) and to not to repeat code everytime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
def fit_predict_show_performance(classifier, X_train, y_train):
    '''
    this function will fit on data passed in argument then it will predict on
    X_test datasetand then will calculate the 10 fold CV accuracy score and then will 
    generate classification report and confusion matrix based on prediction and y_test
    it will only print result, to get all calculated result, uncomment last line and 
    call it like below example:
    y_pred, cr, cm = fit_predict_show_performance(churn_classifier, X_train, y_train)
    '''
    # fitting model
    classifier.fit(X_train, y_train)
    churn_prediction = classifier.predict(X_test)
    # getting K-fold CV scores for K = 10
    ten_performances = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
    k_fold_performance = ten_performances.mean()
    print("K-fold cross validation score of model for k = 10 is :")
    print(k_fold_performance)
    print("====================================")
    print("====== Classification Report ======= ")
    cr = classification_report(y_test,churn_prediction)
    print(cr)
    print("====== Confusion matrix ======= ")
    cm = confusion_matrix(y_test,churn_prediction)
    print(cm)
    #return [churn_prediction, cr, cm]

#########################
#  Logistic Regression  #
#########################
    
# Building Logistic Regression for churn_data_df i.e. with outliers
from sklearn.linear_model import LogisticRegression
churn_classifier = LogisticRegression()
fit_predict_show_performance(churn_classifier, X_train, y_train)

# Building Logistic Regression for churn_data_df_wo i.e. without outliers
churn_classifier = LogisticRegression()
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)


#########################
#       KNN             #
#########################
# knn for churn_data_df i.e. dataset with outliers
from sklearn.neighbors import KNeighborsClassifier
churn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p =2)
fit_predict_show_performance(churn_classifier, X_train, y_train)

# knn for churn_data_df_wo i.e. dataset without outliers
churn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p =2)
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

#########################
#     Naive Bayes       #
#########################
# Naive bayes with outlier i.e. churn_data_df
from sklearn.naive_bayes import GaussianNB
churn_classifier = GaussianNB()
fit_predict_show_performance(churn_classifier, X_train, y_train)

# Naive bayes without outlier i.e. churn_data_df_wo
churn_classifier = GaussianNB()
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

#########################
#    Decision Tree      #
#########################
# Decision tree classifier for churn_data_df with outliers
from sklearn.tree import DecisionTreeClassifier
churn_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=1)
fit_predict_show_performance(churn_classifier, X_train, y_train)

# Decision tree classifier for churn_data_df_wo without outliers
churn_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=1)
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

#########################
#    Random Forest      #
#########################
# Random forest model on churn_data_df i.e. with outliers
from sklearn.ensemble import RandomForestClassifier
churn_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                          random_state=1)
fit_predict_show_performance(churn_classifier, X_train, y_train)

# Random forest model on churn_data_df_wo i.e. without outliers
churn_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                         random_state=1)
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

############################################
#                                          #
#                                          #
#        Hyperparameter tuning             #
#                                          #
#                                          #
############################################
#########################################
#                                       #
# tuning decision tree for both dataset #
# churn_data_df and churn_data_df_wo    #
#                                       #
#########################################

# hyperparameter tuning for Decision tree classifier
from sklearn.model_selection import GridSearchCV
churn_classifier = DecisionTreeClassifier(random_state=1)
params = [{'criterion':['entropy', 'gini'],
          'max_depth':[6,8,10,12,20],'class_weight':['balanced',{0:0.45, 1:0.55}, 
                      {0:0.55,1:0.45},{0:0.40,1:0.60}],'random_state' :[1]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)

# tuning Decision Tree for dataset with outlier i.e. churn_data_df
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_

#Decision tree classifier for churn_data_df i.e. with outliers after tuning parameter
#from sklearn.tree import DecisionTreeClassifier
churn_classifier = DecisionTreeClassifier(criterion = 'entropy', 
                                          class_weight={0:0.55, 1:0.45},max_depth=8,
                                          random_state=1)
fit_predict_show_performance(churn_classifier, X_train, y_train)

# hyperparameter tuning for Decision tree classifier for dataset without outliers
from sklearn.model_selection import GridSearchCV
churn_classifier = DecisionTreeClassifier(random_state=1)
params = [{'criterion':['entropy', 'gini'],
          'max_depth': [6, 8, 10, 12], 'class_weight':['balanced', {0:0.45, 1:0.55},
                       {0:0.55, 1:0.45}, {0:0.40, 1:0.60}], 'random_state' :[1]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_train_wo, y_train_wo)
grid_search.best_params_

# Decision tree classifier for churn_data_df_wo without outliers
churn_classifier=DecisionTreeClassifier(criterion = 'gini', max_depth = 8,
                                        class_weight={0:0.45,1:0.55},random_state=1)
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

############ Hyperparameter tuning for Random Forest #############
# Grid search for finding best parameter for random_forest on churn_data_df dataset
churn_classifier = RandomForestClassifier(random_state=1)
params=[{'criterion':['entropy', 'gini'],'n_estimators':[800,1000],
         'max_depth': [8, 10, 12], 'class_weight':['balanced', {0:0.45, 1:0.55},
                      {0:0.55, 1:0.45}],'random_state' :[1]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_

# tuned randomforest model on chrun_data_df
churn_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',
                                          class_weight='balanced',max_depth=10,
                                          random_state=1)
fit_predict_show_performance(churn_classifier, X_train, y_train)

# tuning on chrun_data_df_wo dataset for random forest model
churn_classifier = RandomForestClassifier(random_state=1)
params = [{'criterion':['entropy', 'gini'],'n_estimators':[600, 800, 1000],
          'max_depth': [8, 10, 12, 14], 'class_weight':['balanced', {0:0.45, 1:0.55}],
           'random_state' :[1]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_train_wo, y_train_wo)
grid_search.best_params_

# tuned Random forest model on churn_data_df_wo i.e. without outliers
churn_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',
                                          max_depth=10,class_weight='balanced',
                                          random_state=1)
fit_predict_show_performance(churn_classifier, X_train_wo, y_train_wo)

############################################
#                                          #
#    SMOTE + Tomek (Oversampling)          #
#         Balancing Target                 #
#                                          #
############################################

# resmapling data from churn_data_df i.e. withoutliers
from imblearn.combine import SMOTETomek
smt = SMOTETomek()
X_resampled, y_resampled = smt.fit_sample(X_train, y_train)

# checking shape of data after resampling
print(X_resampled.shape)
print(y_resampled.shape)
print("class proportion")
print(pd.Series(y_resampled).value_counts(normalize = True))

# Tuning Random Forest model for resampled data from churn_data_df
churn_classifier = RandomForestClassifier(random_state=1)
params = [{'criterion':['entropy', 'gini'],'n_estimators':[600, 800, 1000],
          'max_depth': [20, 22, 24, 26], 'random_state' :[1],
          'class_weight':['balanced', {0:0.55, 1:0.45},{0:0.45, 1:0.55}]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_resampled, y_resampled)
grid_search.best_params_

# building  Random Forest model on tuned hyperparameter
churn_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',
                                          class_weight='balanced',max_depth=24,
                                          random_state=1)
fit_predict_show_performance(churn_classifier, X_resampled, y_resampled)

# resampling data for dataset churn_data_df_wo i.e. without outliers
smt = SMOTETomek()
X_resampled_wo, y_resampled_wo = smt.fit_sample(X_train_wo, y_train_wo)

# checking shape of data
print(X_resampled_wo.shape)
print(y_resampled_wo.shape)
print("class proportion")
print(pd.Series(y_resampled_wo).value_counts(normalize = True))

# tuning Random forest model for resampled data without outliers
churn_classifier = RandomForestClassifier(random_state=1)
params = [{'criterion':['entropy','gini'],'n_estimators':[600, 800, 1000],
          'max_depth': [24, 26, 28], 'random_state' :[1],
          'class_weight':['balanced', {0:0.45, 1:0.55},{0:0.55, 1:0.45}]}]
grid_search = GridSearchCV(estimator=churn_classifier, param_grid=params,
                          scoring = 'f1', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_resampled_wo, y_resampled_wo)
grid_search.best_params_

# building Random Forest model on tuned parameter
churn_classifier = RandomForestClassifier(n_estimators = 800, criterion = 'entropy',
                                          class_weight={0:0.55, 1:0.45},
                                          max_depth=26,random_state=1)
fit_predict_show_performance(churn_classifier, X_resampled_wo, y_resampled_wo)



