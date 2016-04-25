
## The code below was used to ensure all packages are up to date
update.packages(checkBuilt = TRUE)

## packages to install include;
## psych - to gain access to describe function for exploratory data analysis of dataset
## missforest - for imputing missing dataset with mixed continuous and categorical variables
## caret - for logistic regression modelling among other predictive analytics features
## broom - this has tidy() and glance() functions that help to construct a dataframe that summarises the model's statistical findings
## ROCR - to be used for evaluation of performance metrics

library(psych)
## import data into r from csv file
credscor.wimiss <- read.csv("D:/user/Documents/BNP Project Work/Credit Scoring/hmeq-credit-scoring(2).csv")

## the code below is used to describe and summarise the dataset
describe(credscor.wimiss)
summary(credscor.wimiss)

## explore structure of the dataset
str(credscor.wimiss)

## recode target variable to a factor so as to explore the split of values
credscor.wimiss$BAD <- factor(credscor.wimiss$BAD,levels = c("0", "1"), labels = c("Non-Default", "Default"))
str(credscor.wimiss$BAD)
count(credscor.wimiss$BAD)
##   x           freq
## Non-Default   4771
##    Default  1189


## From the count of default vs non-default we can see that the data set has about 4x more non-defaulters than it has defaulters
## It is important to take note of this as it will have an impact on the results of later training models

## plot to visualise defaults vs non-defaults
plot(credscor.wimiss$BAD, title(main = "Split of Default / Non-Default", ylab = "Frequency"))

## re-code bad variable values back to 0 for non-defaulters and 1 for defaulters
credscor.wimiss$BAD <- factor(credscor.wimiss$BAD,levels = c("Non-Default", "Default"), labels = c("0","1"))
str(credscor.wimiss)

str(credscor.wimiss)
## show data types of each variable
sapply(credscor.wimiss, class)

## clean data set to remove rows with missing values
cleancred.data <- na.omit(credscor.wimiss)
str(cleancred.data)


write.csv(cleancred.data, 'D:/user/Documents/BNP Project Work/Credit Scoring/credscore_nomiss.csv', row.names=T)


## Import the example data file(s).
## we can choose to analyse the data set with rows missing values deleted.
## This script only analyses the full dataset with full missing data imputations using missforest

no.miss <- read.table("D:/user/Documents/BNP Project Work/Credit Scoring/credscore_nomiss.csv",header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
summary(no.miss)  # rows with missing values deleted.

## in the code below the na.strings argument has been written in such a way that empty cells with space characters are also changed to NA
wi.miss <- read.table("D:/user/Documents/BNP Project Work/Credit Scoring/hmeq-credit-scoring(2).csv",header=TRUE, sep=",", na.strings= c("", " "), dec=".", strip.white=TRUE)
summary(wi.miss)  # Same data, but with missing values.
str(wi.miss)
## How much missing? 

ncol(wi.miss); nrow(wi.miss)
length(which(is.na(wi.miss) == "TRUE")) / (nrow(wi.miss)*ncol(wi.miss))

## the missforest package within r can be used to impute missing values and is particularly useful in cases where we have mixed type data
## the package can be used to impute continuous and categorical variables including complex interactions and non-linear relations
## Load the necessary package;
## and dependencies: randomForest, foreach, itertools, iterators.

library(missForest)

## Apply the 'missForest' function with all arguments set to default values.
## The function returns a list object with 3 elements: "ximp" which is the
## imputed data, "OOBerror" which is the estimated imputation error, and
## "error" which is the true imputation error. 
## Please note, the function does accept a data frame; the package documentation
## states that the data must be in a matrix (all numeric), however that is not
## the case.


## Please note: large data sets will require considerable more time.

################################################################################
## Now we try an example using the parallel processing argument to utilize
## two cores (on a two core machine). Note, we first need to load an additional
## package (doParallel; and its dependency: parallel) and then register
## the number of processors (i.e. cores).

library(doParallel)
registerDoParallel(cores = 2)

## Now we can apply the 'missForest' function while braking the work down into
## equal numbers of 'variables' or 'forests' for each core to work on (here we
## break the number of variables).

system.time(im.out <- missForest(xmis = wi.miss, maxiter = 10, ntree = 100,variablewise = TRUE,decreasing = FALSE, verbose = FALSE,mtry =  ncol(wi.miss)-1, replace = TRUE,classwt = NULL, cutoff = NULL, strata = NULL,sampsize = NULL, nodesize = NULL, maxnodes = NULL,xtrue = NA, parallelize = "variables"))
credscor.imp <- im.out$ximp  # Extracting only the imputed data from the output list.

## you can use the two lines of code below to compare the original dataset with the imputed dataset
summary(wi.miss)
summary(credscor.imp)
str(credscor.imp)
credscor <- credscor.imp

## the two error values can be computed if there is a complete dataset to compare the imputed one with
credscor$OOBerror
credscor$error

write.csv(credscor, 'D:/user/Documents/BNP Project Work/Credit Scoring/credscore_imputed.csv', row.names=T)

## Before training the model it is important to specify categorical features as factors if they are not already coded as such
credscor$REASON = factor(credscor$REASON)
credscor$JOB = factor(credscor$JOB)
credscor$BAD =factor(credscor$BAD)
table(credscor$BAD)

library(caret)

## I needed the code below to make the caret package work as the package wouldn't work properly without pbkrtest
install.packages("lme4")
packageurl <- "https://cran.r-project.org/src/contrib/Archive/pbkrtest/pbkrtest_0.4-4.tar.gz" 
install.packages(packageurl, repos=NULL, type="source")
############################################################################

credscor.numeric <- subset(credscor.train, select = -c(BAD,REASON,JOB))

str(credscor.numeric)
ncol(credscor.numeric)
descrCorr <- cor(credscor.numeric)
highCorr <- findCorrelation(descrCorr, 0.80)
highCorr
credscor.inputnum <- credscor.numeric[, -highCorr]
testDescr <- testDescr[, -highCorr]
ncol(credscor.inputnum)




set.seed(266)
credscor.sampling_vector <- createDataPartition(credscor$BAD, p=0.75, list = FALSE)

## comma after the square brackets in the formula below ensures that the input variables are included
credscor.train <- credscor[credscor.sampling_vector,]
str(credscor.train)

credscor.train.labels <- credscor$BAD[credscor.sampling_vector]
credscor.test <- credscor[-credscor.sampling_vector,]
credscor.test.labels <- credscor$BAD[-credscor.sampling_vector]

## logistic regression model can be accessed using the glm() function below
credscor.train.model <- glm(BAD ~ ., family = binomial(link = 'logit'), data = credscor.train)

## A summary of the model can be obtained using the summary function
logit.model.summary <- summary(credscor.train.model)
logit.model.summary
## AIC: 3103

## The results show that with the exception of the YOJ variable, 
## all the other continuous input variables are good predictors of the output variable
## For the categorical variables one out of six occupational categories of the JOB variable is a good predictor
## Under the Reason variable one category is a good predictor while the other has been dropped from the model
## The higher the absolute value of the z-statistic, the more likely it is that the feature is significantly related to the output variable
## It is important to note that although variables like YOJ that have high p-values and hence can be interpreted as not adding much to the model
## such features might be good predictors of the output variable in the absence of the other input features 

##To illustrate let's model with only one input variable (YOJ)
YOJ.model <- glm(BAD ~ YOJ, data = credscor.train, family = binomial("logit"))
summary(YOJ.model)
## AIC: 4114.9

## Note the AIC score generated is ameasure of fit that penalises for the number of parameters p.
## Smaller AIC values indicate better fit, hence the AIC can be used to compare models
## In the summary statistics we will observe that the AIC value of the simpler model with only one input variable,
## is higher than that obtained with the full model so we would expect this simple model to be worse.


## write summary to csv file
## Before doing this we need to download the broom package which has tidy() and glance() functions that help to construct
## a dataframe that summarises the model's statistical findings
library(broom)
logit.model.summary <- tidy(credscor.train.model)
logit.glance.credscor <- glance(credscor.train.model)

write.csv(logit.model.summary, 'D:/user/Documents/BNP Project Work/Credit Scoring/credscore_logregsumry.csv')
write.csv(logit.glance.credscor,'D:/user/Documents/BNP Project Work/Credit Scoring/credscore_logregglance.csv')


## testing set performance
## The predict() within the caret package can be used to compute the output of our model.
## This output is the probability of the input belonging to class 1
## A threshold of 0.5 has been chosen. You can experiment with different thresholds

train_predictions <- predict(credscor.train.model, newdata = credscor.train, type = "response")
train_class_predictions <- as.numeric(train_predictions > 0.5)
mean(train_class_predictions == credscor.train$BAD)


test_predictions = predict(credscor.train.model, newdata = credscor.test, type = "prob")
test_class_predictions = as.numeric(test_predictions > 0.5)
mean(test_class_predictions == credscor.test$BAD)




## We can view the confusion matrix and measures to evaluate performance using the code below
cm <- confusionMatrix(train_class_predictions, credscor.train$BAD, positive = "1")
cm


##Confusion Matrix and Statistics

##               Reference
##Prediction       0    1
##    0           3477  553
##    1            102  339

##Accuracy : 0.8535          
##95% CI : (0.8428, 0.8637)
##No Information Rate : 0.8005          
##P-Value [Acc > NIR] : < 2.2e-16       

##Kappa : 0.4339          
##Mcnemar's Test P-Value : < 2.2e-16       

## Sensitivity : 0.38004         
## Specificity : 0.97150         
## Pos Pred Value : 0.76871         
## Neg Pred Value : 0.86278         
## Prevalence : 0.19951         
## Detection Rate : 0.07582         
## Detection Prevalence : 0.09864         
## Balanced Accuracy : 0.67577         

##'Positive' Class : 1               


## if we were to choose a different threshold (different from the initial 0.5 chosen), the preceding metrics would change 
## The ROC curve can be used To visually assess the impact of changing the the threshold on our performance metrics 
library(ROCR)
train_predictions <- predict(credscor.train.model, newdata = credscor.train, type = "response")
pred <- prediction(train_predictions, credscor.train$BAD)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") ## other inputs for the "measure" and "x.measure" arguments are prec and rec respectively
plot(perf, main = "ROC curve for Loan Defaults", col = "green",lwd = 3) ## the argument lwd is used for adjusting line width

## The code below is used to add a reference line to indicate the performance of a classifier with no predictive value
## It produces a diagonal line from bottom left to top-right corner implying the classifier detects true positives and false positives at the same rate
abline(a = 0, b = 1, col = "grey", lwd = 2, lty = 2)

## The code below is used to calculate the area under the ROC curve (AUC) statistic
## The stat ranges from 0.5 to 1 with a 0.5 value depicting a useless classifier, and 1 depicting a perfect classifier
perf.auc <- performance(pred, measure = "auc")
unlist(perf.auc@y.values)

## AUC score of the logistic model here is 0.8235547 which suggests that it is a good classifier

## However based on the sensitivity value of 0.38, this model is not good for predicting defaulters, 
## i.e. there might be a high number of false negatives if the model is used.
## A problem with the model might stem from the fact that only about 20% of observations were cases of default
## perhaps if there is a better representation of defaults in the data then the model will improve.


