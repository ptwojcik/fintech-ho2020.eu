[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MVAparcoo1** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet : fintech-svm

Description : 'Shows sample application of Support Vector Machine on two sets of artificial data.'

Keywords : 'SVM, linear kernel, radial basis kernel, parameter tuning, cross-validation'

Author : Piotr Wójcik

Submitted : Fri, March 29th by Piotr Wójcik

Datafile : data1.Rdata, data2.RData

```

### R Code:
```r

# lets load the needed packages

install.packages("kernlab")
library(kernlab)

library(verification)

library(ggplot2)
library(tidyverse)
library(caret)

#---------------------------------------------------
# lets load the data with two groups
# that are not linearly separable

load("data1.RData")

head(data1)

table(data1$group)

# check it on the plot

ggplot(data = data1, 
       aes(x = X1, y = X2)) +
  geom_point(aes(col = group),
             size = 3) +
  theme_bw()

# let's divide the data randomly
# into train and test samples 

set.seed(987654321)
which_train <- createDataPartition(y = data1$group,
                                   p = 0.7,
                                   list = FALSE)

data1_train <- data1[which_train,]
data1_test <- data1[-which_train,]


# and use the train() function from caret
# to train the model

# we start with no cross-validation

ctrl_nocv <- trainControl(method = "none")

# model is trained on the train sample
# initially with the linear kernel function
# method = "svmLinear"

data1.svm_Linear1 <- train(group ~ X1 + X2, 
                           data = data1_train, 
                           method = "svmLinear",
                           trControl = ctrl_nocv)
# check the result

data1.svm_Linear1

# one can see it also on the classification plot

plot(data1.svm_Linear1$finalModel)

# the distance from the margin
# does not differ much - looks
# like this is a weak classifier

# what is the default value 
# of the cost parameter C?

data1.svm_Linear1$finalModel

# here C = 1 (default for the linear kernel)

# lets's generate forecasts

data1_test$fore_svm_Linear1 <- predict(data1.svm_Linear1, 
                                       newdata = data1_test)

head(data1_test)

table(data1_test$fore_svm_Linear1)

# all observations predicted as group 2!

# confusion matrix

confusionMatrix(data1_test$fore_svm_Linear1, # forecasts 
                data1_test$group, # real values
                # here it does not matter which
                # group is treated as positive 
                # lets take the 1st
                positive = "1") 

# very bad...


# try different values of C
# with repeated cross-validation 

ctrl_cv5x3 <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3)

# the grid of parameters has to be defined
# in the data.frame which column name(s)
# are the same as parameters 
# of the algorithm

parametersC <- data.frame(C = c(0.001, 0.01, 0.02, 0.05, 
                                0.1, 0.2, 0.5, 1, 2, 5))

set.seed(987654321)
data1.svm_Linear2 <- train(group ~ X1 + X2, 
                           data = data1_train, 
                           method = "svmLinear",
                           # parameters are forwarded
                           # to the validation procedure
                           # with the argument tuneGrid=
                           tuneGrid = parametersC,
                           trControl = ctrl_cv5x3)

data1.svm_Linear2

# the value of 0.001 is indicated as optimal,
# but the acuraccy is identical for all C

# one can see it also on the classification plot

plot(data1.svm_Linear2$finalModel)

# same as before...


# one can check the forecasts again

data1_test$fore_svm_Linear2 <- predict(data1.svm_Linear2, 
                                       newdata = data1_test)

confusionMatrix(data1_test$fore_svm_Linear2,
                data1_test$group,
                positive = "1")

# equally weak as for C = 1

# let's swich to radial basis kernel (gaussian)
# method = "svmRadial" - at the beginning 
# without the cross validation

data1.svm_Radial1 <- train(group ~ X1 + X2, 
                           data = data1_train, 
                           method = "svmRadial",
                           trControl = ctrl_nocv)
# the result

data1.svm_Radial1$finalModel

plot(data1.svm_Radial1$finalModel)

# looks better - distances are now very
# diversified and seem to identify
# the shape of groups well

# this algorithm has two parameters:
# - cost C (here by default C = 0.25)
# and sigma - smoothing parameter for
# the radial basis kernel

# the list of parameters required
# by the particular algorithm
# can be checked with the 
# function modelLookup()

modelLookup("svmRadial")

# let's compare the forecasts

data1_test$fore_svm_Radial1 <- predict(data1.svm_Radial1, 
                                       newdata = data1_test)

table(data1_test$fore_svm_Radial1)

# confusion matriz for the test sample 

confusionMatrix(data1_test$fore_svm_Radial1,
                data1_test$group,
                positive = "1")

# radical improvement!!!

# but is this the optimal set of hyperparameters?

# one can check this with cross-validation

# if there is more than one parameter required
# by the algorithm one can easily generate ALL
# unique combinations into a data.frame 
# with the expand.grid() function

# below 6 values of C and 5 sigmas are combined

parametersC_sigma <- 
  expand.grid(C = c(0.01, 0.05, 0.1, 0.5, 1, 5),
              sigma = c(0.05, 0.1, 0.2, 0.5, 1))

head(parametersC_sigma)

nrow(parametersC_sigma)

set.seed(987654321)
data1.svm_Radial2 <- train(group ~ X1 + X2, 
                           data = data1_train, 
                           method = "svmRadial",
                           tuneGrid = parametersC_sigma,
                           trControl = ctrl_cv5x3)

# it may take several seconds

data1.svm_Radial2

plot(data1.svm_Radial2$finalModel)

# also looks good

# optimal values: sigma = 0.5, C = 0.1

# then forecasts are generated

data1_test$fore_svm_Radial2 <- predict(data1.svm_Radial2, 
                                       newdata = data1_test)

table(data1_test$fore_svm_Radial2)

# now both groups predicted,
# but how correct is it?

# confusion matrix on the test sample

confusionMatrix(data1_test$fore_svm_Radial2, 
                data1_test$group,
                positive = "1")

# best out of all compared so far

#-----------------------------------------------------------
# we can check the same on even 
# more complex spiral data

load("data2.RData")

head(data2)

table(data2$group)

# check it on the plot

ggplot(data = data2, 
       aes(x = X1, y = X2)) +
  geom_point(aes(col = group),
             size = 3) +
  theme_bw()

# looks messy, but there is
# a spiral structure behind

# let's divide the data randomly
# into train and test samples 

set.seed(987654321)

which_train <- createDataPartition(y = data2$group,
                                   p = 0.7,
                                   list = FALSE)

data2_train <- data2[which_train,]
data2_test <- data2[-which_train,]

# for sure the data is not linearly separable
# so let's use the radial basis kernel 
# and optimize on the same set 
# of parameters as before

parametersC_sigma <- 
  expand.grid(C = c(0.01, 0.05, 0.1, 0.5, 1, 5),
              sigma = c(0.05, 0.1, 0.2, 0.5, 1))

set.seed(987654321)

data2.svm_Radial1 <- train(group ~ X1 + X2, 
                           data = data2_train, 
                           method = "svmRadial",
                           tuneGrid = parametersC_sigma,
                           trControl = ctrl_cv5x3)

# it may take several seconds

plot(data2.svm_Radial1$finalModel)

# does not seem very good

data2.svm_Radial1

# confusion matrix on the test sample

confusionMatrix(predict(data2.svm_Radial1, 
                        newdata = data2_test), 
                data2_test$group,
                positive = "1")

# does not look impressive

# the optimal cost parameter
# C = 1 and sigma also 1
# which the upper limit of the set

# lest check different parameters
# increasing both the cost and sigma

parametersC_sigma2 <- 
  expand.grid(C = c(1, 5, 10, 25, 50, 100),
              sigma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000))


set.seed(987654321)

data2.svm_Radial2 <- train(group ~ X1 + X2, 
                           data = data2_train, 
                           method = "svmRadial",
                           tuneGrid = parametersC_sigma2,
                           trControl = ctrl_cv5x3)

data2.svm_Radial2

# optimal sigma = 10 and C = 50

plot(data2.svm_Radial2$finalModel)

# looks promising

# confusion matrix on the test sample

confusionMatrix(predict(data2.svm_Radial2, 
                        newdata = data2_test), 
                data2_test$group,
                positive = "1")

# much better !

```
