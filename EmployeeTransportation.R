#All necessory libraries have been pre-loaded.
library(dplyr)
library(forcats)
library(mice)
library(corrplot)
library(psych)
library(car)
library(caTools)
library(ROCR)
library(DMwR)
library(e1071)
library(class)
library(caret)
library(gbm)
library(xgboost)
library(data.table)
library(scales)
library(ineq)
library(ipred)
library(rpart)

#EDA
setwd("/Users/saleesh/Desktop/R Programming")
Transportation=read.csv("carsedited.csv")

View(Transportation)
head(Transportation)
str(Transportation)
summary(Transportation)


#Identify missing values
is.na(Transportation)
summary(is.na(Transportation))
Transportation=na.omit(Transportation)

#Variable treating

Caruse=ifelse(Transportation$Transport == "Car", 1, 0)
Caruse=as.factor(Caruse)
Transportation=cbind(Transportation,Caruse)
Engineer=as.factor(Engineer)
MBA=as.factor(MBA)
license=as.factor(license)

#Identify outliers
boxplot(Transportation)
boxplot(Transportation$Age, main="Age")
boxplot(Work.Exp, main="Work.Exp")
boxplot(Salary, main="Salary")
boxplot(Distance, main="Distance")

#Univariate analysis
hist(Transportation$Age)
hist(Work.Exp)
hist(Salary)
hist(Distance)
plot(Transportation$Gender, main="Gender")
plot(Engineer, main="Engineer")
plot(MBA, main="MBA")
plot(Transportation$Caruse, main="Caruse")

#Bivariate analysis
plot(Transportation$Age,Work.Exp)
cor(Transportation$Age,Work.Exp)
plot(Transportation$Age,Salary)
cor(Transportation$Age,Salary)
plot(Transportation$Age,Distance)
cor(Transportation$Age,Distance)
plot(Work.Exp,Salary)
cor(Work.Exp,Salary)
plot(Work.Exp,Distance)
cor(Work.Exp,Distance)
plot(Salary,Distance)
cor(Salary,Distance)

#Checking Multicolinearity and treating it

glm.trans.full=glm(Caruse~Age+Work.Exp+Salary+Distance+Gender+Engineer+MBA+license,data=Transportation,family="binomial")
vif(glm.trans.full)
summary(glm.trans.full)

cor(Transportation[,-c(2,3,4,8,9,10)])
ev = eigen(cor(Transportation[,-c(2,3,4,8,9,10)]))
ev
EigenValue=ev$values
EigenValue
Factor=c(1,2,3,4)
Scree=data.frame(Factor,EigenValue)
plot(Scree,main="Scree Plot", col="Blue")
lines(Scree,col="Red")

chisq.test(Caruse,Gender)
chisq.test(Caruse,Engineer)
chisq.test(Caruse,MBA)
chisq.test(Caruse,licence)
chisq.test(Gender,Engineer)
chisq.test(Gender,MBA)
chisq.test(Gender,licence)
chisq.test(Engineer,MBA)
chisq.test(Engineer,licence)
chisq.test(MBA,licence)


#Transportation=Transportation[,-c(6,9)]

attach(Transportation)

#Treating data for Analysis

table(Caruse)
set.seed(123)
spl = sample.split(Transportation$Caruse, SplitRatio = 0.7)
train = subset(Transportation, spl == T)
test = subset(Transportation, spl == F)

dim(train)
dim(test)
prop.table(table(train$Caruse))
prop.table(table(test$Caruse))


#Logistic Regression

LRmodel = glm(Caruse ~., data = train, family = binomial)
summary(LRmodel)
vif(LRmodel)

# Eliminating the insignificant variables
# Gender, Engineer, MBA, Salary

train.sub = train [-c(2, 3, 4, 6)]
test.sub = test [-c(2, 3, 4, 6)]

LRmodel.sub = glm(Caruse ~., data = train.sub, family = binomial)
summary(LRmodel.sub)

predTest = predict(LRmodel.sub, newdata = test.sub, type = 'response')

cmLR = table(test.sub$Caruse, predTest>0.1)
sum(diag(cmLR))/sum(cmLR)
cmLR

ROCRpred = prediction(predTest, test$Caruse)
as.numeric(performance(ROCRpred, "auc")@Caruse.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf)

plot(perf, colorize =T, print.cutoffs.at= seq(0, 1, .1), text.adj = c(-.2, 1.7))

set.seed(500)
summary(Transportation$Caruse)
# The data that we have is undersampled 

balanced.data = SMOTE(Caruse ~.,perc.over = 500 , Transportation, k = 5, perc.under = 200)
table(balanced.data$Caruse)

set.seed(144)
split = sample.split(balanced.data$Caruse, SplitRatio = 0.7)
train.bal = subset(balanced.data, split == T)
test.bal = subset(balanced.data, split == F)

LR.smote = glm(Caruse ~., data = train.bal, family = binomial)
summary(LR.smote)

pred.Test.smote = predict(LR.smote, newdata = test.bal, type = 'response')
cm.smote = table(test.bal$Caruse, pred.Test.smote >0.1)
cm.smote
sum(diag(cm.smote))/sum(cm.smote)

#NaiveBayes

NBmodel = naiveBayes(Caruse ~., data = train)
NBpredTest = predict(NBmodel, newdata = test)
tabNB = table(test$Caruse, NBpredTest)
tabNB
sum(diag(tabNB))/sum(tabNB)#Shows 96.2% sensitivity

NBmodel.bal = naiveBayes(Caruse ~., data = train.bal)
NBpredTest.bal = predict(NBmodel.bal, newdata = test.bal)
tabNB.bal = table(test.bal$Caruse, NBpredTest.bal)
tabNB.bal
sum(diag(tabNB.bal))/sum(tabNB.bal)#Shows 94.5% sensitivity


#KNN

train.num = train[,sapply(train, is.numeric)]
test.num = test[,sapply(train, is.numeric)]
names(train.num)
predKNNmodel = knn(train = train.num, test = test.num, cl = train[,3], k = 3)
tabKNN = table(test$Caruse, predKNNmodel)
tabKNN 
sum(diag(tabKNN))/sum(tabKNN)

train.num.bal = train.bal[,sapply(train.bal, is.numeric)]
test.num.bal = test.bal[,sapply(train.bal, is.numeric)]
names(train.num.bal)
predKNNmodel.bal = knn(train = train.num.bal, test = test.num.bal, cl = train.bal[,3], k = 6)
tabKNN.bal = table(test.bal$Caruse, predKNNmodel.bal)
tabKNN.bal
sum(diag(tabKNN))/sum(tabKNN)

knn_fit = train(Caruse ~., data = train.bal, method = "knn",
                trControl = trainControl(method = "cv", number = 3),
                tuneLength = 10)
knn_fit

predKNN_fit = predict(knn_fit, newdata = test.bal[,-8], type = "raw")
tabknnfit=table(test.bal$Caruse, predKNN_fit)
tabknnfit
sum(diag(tabknnfit))/sum(tabknnfit)#Shows 96% sensitivity

# Bagging the Data

Transportation.bagging <- bagging(Caruse ~.,
                          data=train,
                          control=rpart.control(maxdepth=5, minsplit=4))

test$pred.caruse <- predict(Transportation.bagging, test)

table(test$Caruse,test$pred.caruse)




#Boosting method using binary categorical variables and all numeric variables

train.bal$Engineer =as.numeric(train.bal$Engineer)
train.bal$MBA = as.numeric(train.bal$MBA)
train.bal$license = as.numeric(train.bal$license)
train.bal$Caruse = as.numeric(train.bal$Caruse)
test.bal$Engineer =as.numeric(test.bal$Engineer)
test.bal$MBA = as.numeric(test.bal$MBA)
test.bal$license = as.numeric(test.bal$license)
test.bal$Caruse = as.numeric(test.bal$Caruse)
train.bal$Caruse[train.bal$Caruse == 1] = 0
train.bal$Caruse[train.bal$Caruse == 2] = 1
test.bal$Caruse[test.bal$Caruse == 1] = 0
test.bal$Caruse[test.bal$Caruse == 2] = 1

features_train = as.matrix(train.bal[,c(1,2,6,7)])
label_train = as.matrix(train.bal[,7])
features_test = as.matrix(test.bal[,c(1,2,6,7)])

XGBmodel = xgboost(
  data = features_train,
  label = label_train,
  eta = .001,
  max_depth = 5,
  min_child_weight = 3,
  nrounds = 10,
  nfold = 5,
  objective = "binary:logistic", 
  verbose = 0,
  early_stopping_rounds = 10 
)

XGBpredTest = predict(XGBmodel, features_test)
tabXGB = table(test.bal$Caruse, XGBpredTest>0.5)
tabXGB
sum(diag(tabXGB))/sum(tabXGB)







