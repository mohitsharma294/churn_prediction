# importing all required library
required_library <- c('ggplot2', 'corrgram', 'corrplot', 'randomForest',
                      'caret', 'class', 'e1071', 'rpart', 'mlr','grid',
                      'DMwR','irace','usdm')

# checking for each library whether installed or not
# if not install then installing it first and then attaching to file
for (lib in required_library){
  if(!require(lib, character.only = TRUE))
  {
    install.packages(lib)
    require(lib, character.only = TRUE)
  }
}

# removing extra variable
rm(required_library,lib)


# Reading train and test csv file
# set working directory to the file location, uncomment below line and put full path
# setwd("full path to folder in which file is present")
churn_data_df <- read.csv("Train_data.csv")
test_data_df <- read.csv("Test_data.csv")

############################################
#                                          #
#     2.1 Exploratory Data Analysis        #
#                                          #
############################################

###################################
#  2.1.1 understanding the data   #
###################################

# Checking columns name, in R there is syntax defined for column name.
# every space is changed into dot (.) and column name can not start with number etc.
colnames(churn_data_df)     # look at columns name, all changed to as per R syntax

# cheking datatypes of all columns
str(churn_data_df)

### checking numerical variables ###
# Checking numerical statistics of numerical columns (Five point summary + mean of all column)
summary(churn_data_df)

### Checking categorical variable ###
# unique values in each category
cat_col <- c('area.code','international.plan', 'voice.mail.plan','Churn')
lapply(churn_data_df[,c('state', cat_col)], function(feat) length(unique(feat)))

# counting of each unique values in categorical columns
lapply(churn_data_df[,cat_col], function(feature) table(feature))


###################################
#  2.1.2 Missing value analysis   #
###################################

# checking missing value for each column and storing counting in dataframe with column name
missing_val <- data.frame(lapply(churn_data_df, function(feat) sum(is.na(feat))))


###################################
#  2.1.3 outlier analysis         #
###################################

# removing phone number column and changing area code to category
churn_data_df$phone.number <- NULL
churn_data_df$area.code <- as.factor(churn_data_df$area.code)

test_data_df$phone.number <- NULL
test_data_df$area.code <- as.factor(test_data_df$area.code)

# taking out list of name of numerical columns in dataset
numeric_columns <- colnames(Filter(is.numeric, churn_data_df))


# box_plot function to plot boxplot of numerical columns
box_plot <- function(column, dataset){
  ggplot(aes_string(x = 'Churn', y = column, fill = 'Churn'),
         data = dataset)+
    stat_boxplot(geom = 'errorbar', width = 0.5)+
    geom_boxplot(outlier.size = 2, outlier.shape = 18)+
    theme(legend.position = 'bottom')+
    labs(y = gsub('\\.', ' ', column), x = "Churn")+
    ggtitle(paste(" Box Plot :",gsub('\\.', ' ', column)))
}

# hist_plot function to plot histogram of numerical variable
hist_plot <- function(column, dataset){
  ggplot(aes_string(column), data = dataset)+
    geom_histogram(aes(y=..density..), fill = 'skyblue2')+
    geom_density()+
    labs(x = gsub('\\.', ' ', column))+
    ggtitle(paste(" Histogram :",gsub('\\.', ' ', column)))
}


# calling box_plot function and storing all plots in a list
all_box_plots <- lapply(numeric_columns,box_plot, dataset = churn_data_df)

# calling hist_plot function and storing all plots in a list
all_hist_plots <- lapply(numeric_columns,hist_plot, dataset = churn_data_df)

# Plotting Boxplot and histogram to analyse the data for three columns simultaneously
plot_in_grid <- function(f, s, t){
gridExtra::grid.arrange(all_box_plots[[f]],all_box_plots[[s]],all_box_plots[[t]],
                        all_hist_plots[[f]],all_hist_plots[[s]],all_hist_plots[[t]],ncol=3,nrow=2)
}

# plotting for day's minute, call and charges
plot_in_grid(3,4,5)

# plotting for evening's minute, call and charges
plot_in_grid(6,7,8)

# plotting for night's minute, call and charges
plot_in_grid(9, 10, 11)

# plotting for international's minute, call and charges
plot_in_grid(12, 13, 14)

# plotting for account length, voice mail message and customer service calls
plot_in_grid(1, 2, 15)


#####################
# outlier removing  #
#####################
# Note: Considering both dataset one with outliers and other withoutliers for building model
# Reason explained in Project report
#
# name of dataset with outlier :- churn_data_df
# name of dataset without outlier:- churn_data_df_wo

churn_data_df_wo <- churn_data_df

# removing numeric columns for which we will not do outlier removal process
numeric_columns1 <- numeric_columns[! numeric_columns %in% c("number.vmail.messages","number.customer.service.calls")]

for (i in numeric_columns1){
  out_value = churn_data_df_wo[,i] [churn_data_df_wo[,i] %in% boxplot.stats(churn_data_df_wo[,i])$out]
  churn_data_df_wo = churn_data_df_wo[which(!churn_data_df_wo[,i] %in% out_value),]
}

# Plotting again distribution and boxplot after outlier removal

# calling box_plot function and storing all plots in a list 
# for churn_data_df_wo i.e. dataset without outliers
all_box_plots <- lapply(numeric_columns,box_plot, dataset = churn_data_df_wo)

# calling hist_plot function and storing all plots in a list
# for churn_data_df_wo i.e. dataset without outliers
all_hist_plots <- lapply(numeric_columns,hist_plot, dataset = churn_data_df_wo)

# plotting for day's minute, call and charges after outlier removal
plot_in_grid(3,4,5)

# plotting for evening's minute, call and charges after outlier removal
plot_in_grid(6,7,8)

# plotting for night's minute, call and charges after outlier removal
plot_in_grid(9, 10, 11)

# plotting for international's minute, call and charges after outlier removal
plot_in_grid(12, 13, 14)

# plotting for account length, voice mail message and customer service calls
# after outlier removal
plot_in_grid(1, 2, 15)


###################################
#  2.1.4 Feature Selection        #
###################################

# correlation plot for numerical feature
corrgram(churn_data_df[,numeric_columns], order = FALSE,
         upper.panel = panel.pie, text.panel = panel.txt,
         main = "Correlation Plot for Churning data set")

# heatmap plot for numerical features
corrplot(cor(churn_data_df[,numeric_columns]), method = 'color', type = 'lower')

# getting categorical column
cat_col <- c('state', 'area.code','international.plan', 'voice.mail.plan')

# chi-square test of independence of each category with Churn column
for(i in cat_col){
  print(i)
  print(chisq.test(table(churn_data_df$Churn, churn_data_df[,i])))
}

# Now checking multicollinearity between international plan and voice mail plan
# by chi-sq test of independence
print(chisq.test(table(churn_data_df$international.plan,
                       churn_data_df$voice.mail.plan)))

# checking VIF factor for numeric columns
vif(churn_data_df[,numeric_columns])

# checking importance of feature in ranking using random forest
important_feat <- randomForest(Churn ~ ., data = churn_data_df,
                               ntree = 200, keep.forest = FALSE, importance = TRUE)
importance_feat_df <- data.frame(importance(important_feat, type = 1))

################################
#                              #
#     Data After EDA           #
#                              #
################################

# Dropping column state, area code as in chi-sq test these column were
# not dependent with Churn column. Dropping  total day min, total eve charge, total night charge,
# total intl charge and these columns found to be multicolinear with other columns
churn_data_df <- churn_data_df[, -c(1,3,7,12,15,18)]
churn_data_df_wo <- churn_data_df_wo[, -c(1,3,7,12,15,18)]
test_data_df <- test_data_df[, -c(1,3,7,12,15,18)]


# checking VIF factor for numeric columns after removal of multicollinear columns
numeric_columns <- colnames(Filter(is.numeric, churn_data_df))
vif(churn_data_df[,numeric_columns])

# changing levels of factor to 0 and 1  
# no :- 0, yes:- 1
# false. :- 0, true. :- 1
category = c('international.plan', 'voice.mail.plan', 'Churn')

for (i in category){
  levels(churn_data_df[,i]) <- c(0,1)
  levels(churn_data_df_wo[,i]) <- c(0,1)
  levels(test_data_df[,i]) <- c(0,1)
}


############################################
#                                          #
#                                          #
#  2.2.2 Building Classification models    #
#                                          #
#                                          #
############################################

################################################
#  K-fold CV accuracy score calculator method  #
################################################

### Creating function which will calculate the K-fold CV accuracy
model.K_fold.accuracy <- function(classifier, data){
  # creating 10 folds of data
  ten_folds = createFolds(data$Churn, k = 10)
  # lapply function will result in 10 accuracy measure for each test fold
  ten_cv = lapply(ten_folds, function(fold) {
    training_fold = data[-fold, ]
    test_fold = data[fold, ]
    
    # changing data of classifier with our training folds
    classifier$data = training_fold
    # predicting on test folds
    # for logisitic regression "glm" we got prediction in probability
    # changing probabiliey to class
    if(class(classifier)[1] == "glm"){
      y_prob = predict(churn_classifier, type = 'response', newdata = test_fold[-14])
      y_pred = ifelse(y_prob>0.5, 1, 0)
    } else if(class(classifier)[1] == 'rpart'){
      y_pred = predict(churn_classifier, newdata = test_fold[-14], type ='class')
    } else{
      y_pred = predict(churn_classifier, newdata = test_fold[-14])
    }
    # creating confusion matrix 
    cm = table(test_fold[, 14], y_pred)
    # calculating accuracy correct prediction divide by all observation
    accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
    return(accuracy)
  })
  # returning mean of all accuracy which we got from lapply function result
  return(mean(as.numeric(ten_cv)))
}

###########################################################
#  Function Predicting result on test data set of a model #
#  And returning confusion matrix                         #
###########################################################

churn.predict <- function(classifier, data){
  if(class(classifier)[1] == 'glm'){
    churn_prob <- predict(classifier, newdata = data[,-14])
    churn_prediction <- ifelse(churn_prob >= 0.5, 1, 0)
  } else if(class(classifier)[1] == 'rpart'){
    churn_prediction = predict(classifier, data[,-14], type ='class')
  } else{
        churn_prediction = predict(classifier, data[,-14])
  }
  cm = confusionMatrix(table(data$Churn, churn_prediction))
  return(cm)
}

#########################
#  Logistic Regression  #
#########################

# logistic regression on dataset churn_data_df with outliers
churn_classifier <- glm(formula = Churn ~ ., family = binomial,
                       data = churn_data_df)
cm <- churn.predict(churn_classifier, test_data_df)
cm

# K -fold accuracy of Logistic regression model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df)
k_fold_accuracy

# Now checking on dataset without ouliers
churn_classifier <- glm(formula = Churn ~ ., family = binomial,
                       data = churn_data_df_wo)
cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Logistic regression model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df_wo)
k_fold_accuracy


#########################
#       KNN             #
#########################

# predicting on dataset with outliers i.e. churn_data_df
churn_prediction <- knn(train = churn_data_df[,-14], test = test_data_df[,-14],
                        cl = churn_data_df$Churn, k = 5, prob = TRUE)
confusionMatrix(table(test_data_df$Churn, churn_prediction))


# predicting on dataset with outliers i.e. churn_data_df_wo
churn_prediction <- knn(train = churn_data_df_wo[,-14], test = test_data_df[,-14],
                        cl = churn_data_df_wo$Churn, k = 5, prob = TRUE)
confusionMatrix(table(test_data_df$Churn, churn_prediction))


#########################
#     Naive Bayes       #
#########################

# Building model on dataset with outliers i.e. churn_data_df
churn_classifier <- naiveBayes(x = churn_data_df[,-14], y =churn_data_df[,14])

cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Naive Bayes model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df)
k_fold_accuracy


# building model on dataset without outliers i.e. churn_data_df_wo
churn_classifier <- naiveBayes(x = churn_data_df_wo[,-14], y =churn_data_df_wo[,14])
cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Naive Bayes model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df_wo)
k_fold_accuracy


#########################
#    Decision Tree      #
#########################

# building model on dataset with outliers i.e. churn_data_df
churn_classifier <- rpart(formula = Churn ~ ., data = churn_data_df)

cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df)
k_fold_accuracy

# building model on dataset without outliers i.e. churn_data_df_wo
churn_classifier <- rpart(formula = Churn ~ ., data = churn_data_df_wo)
cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df_wo)
k_fold_accuracy

#########################
#    Random Forest      #
#########################

# building model on dataset with outliers i.e. churn_data_df
churn_classifier <- randomForest(formula = Churn ~ ., data = churn_data_df,
                                 ntree = 500)

cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df)
k_fold_accuracy

# building model on dataset without outliers i.e. churn_data_df_wo
churn_classifier <- randomForest(formula = Churn ~ ., data = churn_data_df_wo,
                                 ntree = 500)
cm <- churn.predict(churn_classifier, test_data_df)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, churn_data_df_wo)
k_fold_accuracy


############################################

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

# we will tune best two model among above i.e. Decision tree and random Forest
# for tuning we will use mlr package and its methods

tune.Decision.Tree <- function(learner, paramset, dataset){
  # creating task for train 
  train_task = makeClassifTask(data = dataset, target = 'Churn')
  
  # setting 10 fold cross validation
  cv = makeResampleDesc("CV", iters = 10)
  grid_control = makeTuneControlGrid()
  # tuning parameter
  tune_param = tuneParams(learner = learner, resampling = cv, task = train_task,
                          par.set = paramset, control = grid_control, measures = acc)
  return(tune_param)
}

# tuning decision tree classifier for churn_data_df i.e. whole dataset
# making learner tree
learner = makeLearner("classif.rpart", predict.type = 'response')

# setting params range
param_set <- makeParamSet(
  makeIntegerParam("minsplit", lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
) 

tuned_param <- tune.Decision.Tree(learner, param_set, churn_data_df)

# building decision tree model based on tuned param with mlr package
set_param <- setHyperPars(learner, par.vals = tuned_param$x)
train_task <- makeClassifTask(data = churn_data_df, target = 'Churn')
test_task <- makeClassifTask(data = test_data_df, target = 'Churn')
# training model
train_model <- train(set_param, train_task)
# predicting on test data
pred <- predict(train_model, test_task)
y_pred = pred[["data"]][["response"]]
# confusion matrix
cm = table(test_data_df[, 14], y_pred)
cm

# tuning decision tree like above for churn_data_df_wo i.e. without outliers
tuned_param <- tune.Decision.Tree(learner, param_set, churn_data_df_wo)
set_param <- setHyperPars(learner, par.vals = tuned_param$x)
train_task_wo <- makeClassifTask(data = churn_data_df_wo, target = 'Churn')
train_model <- train(set_param, train_task_wo)
pred <- predict(train_model, test_task)
y_pred = pred[["data"]][["response"]]
cm = table(test_data_df[, 14], y_pred)
cm


#########################################
#                                       #
# tuning random forest for both dataset #
#  churn_data_df and churn_data_df_wo   #
#                                       #
#########################################

# tuning random forest for churn_data_df
train_task = makeClassifTask(data = churn_data_df, target = 'Churn')
test_task <- makeClassifTask(data = test_data_df, target = 'Churn')
# making learner
rfLearner <- makeLearner("classif.randomForest", predict.type = 'response',
                         par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(importance = TRUE)
rf_param_set <- makeParamSet(
  makeIntegerParam("ntree",lower = 600, upper = 800),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50))
ctrl <- makeTuneControlIrace(maxExperiments = 200L)
cv <- makeResampleDesc("CV",iters = 3L)
rf_tuned_param <- tuneParams(learner = rfLearner, resampling = cv, task=train_task,
                             par.set = rf_param_set, control = ctrl, measures = acc)
# making model on tuned parameter
rf_set_param <- setHyperPars(rfLearner, par.vals = rf_tuned_param$x)
rf_train_model <- train(rf_set_param, train_task)
pred <- predict(rforest, test_task)
y_pred <- pred$data$response
cm <- table(test_data_df[, 14], y_pred)
cm

###################
# tuning random forest for churn_data_df_wo
train_task_wo <- makeClassifTask(data = churn_data_df_wo, target = 'Churn')
# tuning model
rf_tuned_param <- tuneParams(learner = rfLearner, resampling = cv, task=train_task_wo,
                             par.set = rf_param_set, control = ctrl, measures = acc)
# making model on tuned parameter
rf_set_param <- setHyperPars(rfLearner, par.vals = rf_tuned_param$x)
rf_train_model <- train(rf_set_param, train_task_wo)
pred <- predict(rforest, test_task)
y_pred <- pred$data$response
cm <- table(test_data_df[, 14], y_pred)
cm


############ Alternative way for hyperparameter tuning using Caret #############
# tuning decision tree
control <- trainControl(method="repeatedcv", number=10, repeats=3)
churn_model <- caret::train(Churn ~., data = churn_data_df, method = 'rpart', trControl = control)
churn_model$bestTune
y_pred <- predict(churn_model, test_data_df)
confusionMatrix(test_data_df$Churn, y_pred)
# Note: Different method are available in caret which tune differnt parameter
# see caret documentation for variety

# tuning Random Forest
control <- trainControl(method="repeatedcv", number=10, repeats=3)
churn_model <- caret::train(Churn ~., data = churn_data_df, method = 'rf', trControl = control)
churn_model$bestTune
y_pred <- predict(churn_model, test_data_df)
confusionMatrix(test_data_df$Churn, y_pred)
# for random forest also different methods available, see documentation



############################################
#                                          #
#                                          #
#        SMOTE (Oversampling)              #
#         Balancing Target                 #
#                                          #
############################################
# with the help of caret package we can apply dataset directly by selecting
# sampling = smote

ctrl <- trainControl(method = 'repeatedcv', number = 10,repeats = 10,
                     sampling = 'smote')
set.seed(1)

# smote on churn_data_df and applying on randomForest
rf_model_smote <- caret::train(Churn ~ ., data = churn_data_df, method = 'rf',
                               preProcess = c("scale", "center"),trControl = ctrl)
churn.predict(rf_model_smote, test_data_df)

# smote on churn_data_df_wo and applying on randomForest
rf_model_smote_wo <- caret::train(Churn ~ ., data = churn_data_df_wo, method = 'rf',
                               preProcess = c("scale", "center"),trControl = ctrl)
churn.predict(rf_model_smote_wo, test_data_df)



