#### 0. INCLUDES ####
library(lattice)
library(ggplot2)
library(caret)
library(recipes)
library(C50)
library(corrplot)
library(mlbench)
library(readr)
library(class)
library(dplyr)
Complete<-read.csv("C:/Users/Kiko Sánchez/Desktop/Ubiqum/Course 2/Task 2/datasets/CompleteResponses.csv")
summary(Complete)
str(Complete)
View(Complete)
Incomplete<-read.csv("C:/Users/Kiko SÃ¡nchez/Desktop/Ubiqum/Course 2/Task 2/datasets/SurveyIncomplete.csv")
str(Incomplete)


#### 1. PRE-PROCESSING ####

#Converting numerical to nominal features

#education level
Complete$elevel<-factor(Complete$elevel,
                                   levels = c(0,1,2,3,4),
                                   labels = c("Basic education", "High School", "Some College", "College Degree", "PHD"))
#cars brands
Complete$car<-factor(Complete$car,
                              levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                              labels = c("BMW","Buick", "Cadillac", "Chevrolet","Chrysler","Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln","Mazda","Mercedes","Mitsubishi","Nissan","Ram","Subaru","Toyota","None"))

#regions
Complete$zipcode <- factor(Complete$zipcode,
                                    levels = c(0,1,2,3,4,5,6,7,8),
                                    labels = c("New England","Mid Atlantic","East North Central","West North Central","South Atlantic","East South Central","West South Central","Mountain","Pacific"))
#computer brands
Complete$brand<-factor(Complete$brand,
                                 levels = c(0,1),
                                 labels = c("Acer","Sony"))

#export to csv the 'Complete data set'
write.csv(Complete,"complete_responses_ok.csv")

#### 1.1 feature selection ####

# Correlation matrix (for numerical variables)
?corrplot
#create the numeric matrix
mat_num<-Complete[,c("salary","age","credit")]
#assign the numeric matrix into a correlation matrix (corr)
cor_mat_num<-cor(mat_num)
#plot the results of your correlation matrix (with numbers method)
corrplot(cor_mat_num, method = "number")

#chi square for categorical variables
?chisq.test
chisq.test(Complete$brand, Complete$elevel)


#Anova for numerical and categorical
?aov
aov(salary~brand,Complete)
summary(aov(salary~brand,Complete))
summary(aov(age~brand,Complete))
summary(aov(credit~brand,Complete))


#generalized Linear Modes (for numeric variables)
#glm(formula = X ~ Y, family = binomial(link = "logit"), data = MYDATA)#

glm_salary<-glm(formula =  Complete$brand~Complete$salary,family = binomial(link = "logit"))
summary(glm_salary)

glm_age<-glm(formula =  Complete$brand~Complete$age,family = binomial(link = "logit"))
summary(glm_age)

glm_credit<-glm(formula =  Complete$brand~Complete$credit,family = binomial(link = "logit"))
summary(glm_credit)

#### 2. PLOTTING ####

?ggplot

#Analysing each variable ("salary"  "age"     "elevel"  "car"     "zipcode" "credit"  "brand")

#histogram for each numeric variable (salary, age, credit)
ggplot(Complete, aes(x=age))+
      geom_histogram(color="blue", fill="cyan", bins = 10)

#bars for each categorical variable (elevel, car, zipcode, brand)
ggplot(Complete, aes(x=brand))+
  geom_bar(color="blue", fill="cyan")


#Comparing the dependent variable (fill=brand) with independent ones

#histogram for numericals (and comparing with brand)
ggplot(Complete, aes(x=credit,fill=brand))+
      geom_histogram(color="blue", bins = 10)+
      scale_fill_manual(values = c("#FFFF00", "#00FFFF"))


#scatter plot for numericals (and comparing with brand)

ggplot(Complete,aes(x=age,y=salary,color=brand))+
  geom_point()+
  scale_colour_manual(values = c("Sony" = "cyan", "Acer" = "yellow"))

#bars for nominals/categoricals
ggplot(Complete, aes(x=brand,fill=brand))+
  geom_bar(color="blue", bins = 10)+
  scale_fill_manual(values = c("#FFFF00", "#00FFFF"))

ggplot(Complete, aes(x=elevel,fill=brand))+
  geom_bar(color="blue")+
  scale_fill_manual(values = c("#FFFF00", "#00FFFF"))+
  xlab('Education') +
  ylab('Count')


#### 3. MODELING ####

#creating training and testing sets from Complete
set.seed(28)

inTraining<-createDataPartition(Complete$brand, p=.75, list = FALSE)
training<-Complete[inTraining,]
testing<-Complete[-inTraining,]

#10 fold (number=10) cross validation (repeatedcv)
cvtraincontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#### 3.1 random forest - allvariables ####
#creating the train Random Forest Regression model with .= all the variables (brand~.)
rf_all <- train(brand~., data = training, method = "rf", trControl=cvtraincontrol, tuneLength=5)

#training results from a Random Forest model with all the variables --> mtry:18, Accuracy:0.92, Kappa:0.83
rf_all

#creating the prediction of the model applied to the testing size
Pred_all_rf<-predict(rf_all,testing)

#prediction results applied to testing size from a Random Forest model with all variables
Pred_all_rf
summary(Pred_all_rf)

#comparing prediction with real values --> Accuracy: 0.9147130, Kappa: 0.8189557
postResample(Pred_all_rf,testing$brand)
summary(postResample(Pred_all_rf,testing$brand))

# save the model to disk
saveRDS(Pred_all_rf, "./pred_all_rf.rds")

#### 3.1.1 random forest - only with salary & age ####
#data.frame for manual tuning of the grid --> number of grid = number of variables
rfgrid<-expand.grid(mtry=c(1,2))

#creating the train Random Forest Regression model with salary & age (brand~salary+age)
#system time wrapper. system.time()is used to measure process execution time
system.time(rf_sal_age <- train(brand~salary+age, data = training, method = "rf", trControl=cvtraincontrol, tuneGrid=rfgrid))

#training results from a Random Forest model with salary & age --> mtry:1, Accuracy:0.92, Kappa:0.83
rf_sal_age

#creating the prediction of the model applied to the testing size
Pred_sal_age_rf<-predict(rf_sal_age,testing)

#prediction results applied to testing size from a Random Forest model with salary & age
Pred_sal_age_rf
summary(Pred_sal_age_rf)

#comparing prediction with real values --> Accuracy: 0.92, Kappa: 0.82
postResample(Pred_sal_age_rf,testing$brand)
summary(postResample(Pred_sal_age_rf,testing$brand))

# save the model to disk
saveRDS(Pred_sal_age_rf, "./pred_sal_age_rf.rds")

#### 3.2 c5.0 method - all variables ####
#creating the train C 5.0 model with .= all the variables (brand~.)
c5_all <- train(brand~., data = training, method = "C5.0", trControl=cvtraincontrol, tuneLength=2)

#training results C5.0 model with all the variables 
c5_all

#creating the prediction of the model applied to the testing size
Pred_all_c5<-predict(c5_all,testing)

#prediction results applied to testing size from a Random Forest model with all variables
Pred_all_c5

#comparing prediction with real values --> Accuracy: 0.9147130, Kappa: 0.8189557
postResample(Pred_all_c5,testing$brand)
summary(postResample(Pred_all_c5,testing$brand))


#variable importance
var_imp_c5<- varImp(c5_all, scale=FALSE)
var_imp_c5
plot(var_imp_c5)

# save the model to disk
saveRDS(Pred_all_c5, "./pred_all_c5.rds")

#### 3.2.1 c.5 method - only with salary & age ####

#creating the train Random Forest Regression model with salary & age (brand~salary+age)
#system time wrapper. system.time()is used to measure process execution time
system.time(c5_sal_age <- train(brand~salary+age, data = training, method = "C5.0", trControl=cvtraincontrol, tuneLength=2))

#training results from a Random Forest model with salary & age --> Accuracy:0.92, Kappa:0.84
c5_sal_age

#creating the prediction of the model applied to the testing size
Pred_sal_age_c5<-predict(c5_sal_age,testing)

#prediction results applied to testing size from a Random Forest model with salary & age
Pred_sal_age_c5
summary(Pred_sal_age_c5)

#comparing prediction with real values --> Accuracy: 0.91, Kappa: 0.82
postResample(Pred_sal_age_c5,testing$brand)
summary(postResample(Pred_sal_age_c5,testing$brand))

# save the model to disk
saveRDS(Pred_sal_age_c5, "./pred_sal_age_c5.rds")


#### 3.3 knn - all variables ####
# Normalize (method=range) the dataset between values 0 and 1
normalize2 <- preProcess(Complete[,], method=c("range"))
normalize2

#applying normalize function to a new data set
Complete_norm <- predict(normalize2, Complete)


# creating training and testing size of the normalize data
set.seed(28)

inTraining_norm<-createDataPartition(Complete_norm$brand, p=.75, list = FALSE)
training_norm<-Complete_norm[inTraining_norm,]
testing_norm<-Complete_norm[-inTraining_norm,]

# creating the model

knnFit_all <- train(brand ~ ., training_norm, method = "knn", trControl=cvtraincontrol)
knnFit_all

#### 3.4.1 knn - with salary and age ####
knnFit_sal_age <- train(brand ~ salary+age, data = training_norm, method = "knn", trControl=cvtraincontrol, tuneLength=100)
knnFit_sal_age
plot(knnFit_sal_age)

#### 3.4.3 knn - with salary, age and credit ####
knnFit_sal_age_cred <- train(brand ~ salary+age+credit, data = training_norm, method = "knn", trControl=cvtraincontrol)
knnFit_sal_age_cred

#### 3.4.3 knn - only with salary ####

knnFit_sal <- train(brand ~ salary, data = training_norm, method = "knn", trControl=cvtraincontrol)
knnFit_sal

#### 4. APPLYING THE MODEL ####

#Converting numerical to nominal features

#education level
Incomplete$elevel<-factor(Incomplete$elevel,
                        levels = c(0,1,2,3,4),
                        labels = c("Basic education", "High School", "Some College", "College Degree", "PHD"))
#cars brands
Incomplete$car<-factor(Incomplete$car,
                     levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                     labels = c("BMW","Buick", "Cadillac", "Chevrolet","Chrysler","Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln","Mazda","Mercedes","Mitsubishi","Nissan","Ram","Subaru","Toyota","None"))

#regions
Incomplete$zipcode <- factor(Incomplete$zipcode,
                           levels = c(0,1,2,3,4,5,6,7,8),
                           labels = c("New England","Mid Atlantic","East North Central","West North Central","South Atlantic","East South Central","West South Central","Mountain","Pacific"))
#converting computer brands as labels
Incomplete$brand<-factor(Incomplete$brand,
                       levels = c(0,1),
                       labels = c("Acer","Sony"))

#as we will use KNN model, we have to normalize the 'incomplete data frame'
# Normalize (method=range) the dataset between values 0 and 1
#with the same normalization applied on the training model to don't change values
Incomplete_norm <- predict(normalize2, Incomplete)
Incomplete_norm

#creating the normalized new data set
Incomplete_norm <- predict(normalize2, Incomplete)

#applying the model to new data frame
Incomplete_predict<-predict(knnFit_sal_age,Incomplete_norm)
Incomplete_predict
summary(Incomplete_predict)


#overwriting the brand column of 'Incomplete data set' with the predicted results
Incomplete$brand<-Incomplete_predict

ggplot(Incomplete_norm, aes(x=Incomplete_predict))+
  geom_bar(color="blue", fill="cyan")
  

#Exporting to csv the 'Incomplete data set' with the predicted results
write.csv(Incomplete,"brand_predicted.csv")
