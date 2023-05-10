#Statistics.jl 
#IT327: Julia Project - Intermediate Program
#Christiana Beard
#4/29/23

#Program to show how can be used for data analysis and linear regression
#The guiding research question: "Is there a relationship between sleep and daily activities?"
#We will create a multiple linear regression model to predict total minutes asleep from various daily activity metrics tracked from a fitbit

#to import new packages
#using Pkg
#Pkg.add("Package Name")

############## Import Data ######################################################################
using DataFrames #package for storing data as a dataframe
using CSV #Package since our data will be in a comma separated values file
fb = CSV.read("fitbit.csv", DataFrame)  #fb stands for fitbit. Data is collected from a survey that collected user's fitbit data - before removing outliers
#Data obtained from https://www.kaggle.com/datasets/arashnic/fitbit
#Data altered to contain a single CSV file with variables:
    # Total Steps - number of steps in a day 
    # Total Distance - the miles traveled in a day
    # Calories - number of calories burned in a day 
    # Total Minutes Active - the total minutes a person was very active, moderately active, or lightly active in a day (not sedentary) 
    # Total Sleep - the total minutes of sleep in a night

size(fb) #shows the dimensions of the data 
describe(fb) #descriptive statistics of the data

############## boxplots before removing outliers ######################################################################
using Plots #important plotting package. used for histograms
using StatsPlots #package used for boxplots
#The boxplot method is used to visualize which observations we need to remove in our data
boxplot(fb[!,1], xlabel = "Total Steps With Outliers")
boxplot(fb[!,2], xlabel = "Total Distance With Outliers")
boxplot(fb[!,3], xlabel = "Calories With Outliers")
boxplot(fb[!,4], xlabel = "Total Minutes Active With Outliers")
boxplot(fb[!,5], xlabel = "Total Minutes Asleep With Outliers")

############## Remove Outliers ######################################################################
#remove outliers from our data 
fitbit = fb[(fb.TotalSteps .<=19769) .& (fb.TotalDistance .<=14) .& (fb.Calories .<=4500) .& (fb.TotalMinutesActive .<=446) .& (fb.TotalMinutesActive .>=79) .& (fb.TotalMinutesAsleep .<=658) .& (fb.TotalMinutesAsleep .>=200), :]

size(fitbit) #shows the dimensions of the data 
describe(fitbit) #descriptive statistics of the data

############## Visualize Data ######################################################################
boxplot(fitbit[!,1], xlabel = "Total Steps Without Outliers")
boxplot(fitbit[!,2], xlabel = "Total Distance Without Outliers")
boxplot(fitbit[!,3], xlabel = "Calories Without Outliers")
boxplot(fitbit[!,4], xlabel = "Total Minutes Active Without Outliers")
boxplot(fitbit[!,5], xlabel = "Total Minutes Asleep Without Outliers")

############## histograms ######################################################################
Plots.histogram(fitbit[!,1], xlabel = "Total Steps")
Plots.histogram(fitbit[!,2], xlabel = "Total Distance")
Plots.histogram(fitbit[!,3], xlabel = "Calories")
Plots.histogram(fitbit[!,4], xlabel = "Total Minutes Active")
Plots.histogram(fitbit[!,5], xlabel = "Total Minutes Asleep")

############## Correlation Plots ######################################################################
using Statistics #package needed to make a correlation plot
cols = [1, 2, 3, 4, 5] 
M = cor(Matrix(fitbit[!,cols])) #correlation matrix
#Total distance is removed because it has a shared correlation coefficient greater that the absolute value of 0.7

#remove high correlation
cols_2 = [1, 3, 4, 5] 
M = cor(Matrix(fitbit[!,cols_2])) #correlation matrix
#No more variables have a high correlation coefficient greater than 0.7

############## Modeling ######################################################################
using GLM #package used for linear regression
funct = @formula(TotalMinutesAsleep ~ TotalSteps + Calories + TotalMinutesActive)
linear_model = glm(funct, fitbit, Normal(), IdentityLink())

#remove variables that are insignificant
#The Calories variable is removed because it had a p-value greater than 0.1. 
#Calories is not significant significant in our linear regression model
funct_2 = @formula(TotalMinutesAsleep ~ TotalSteps + TotalMinutesActive)
linear_model_2 = glm(funct_2, fitbit, Normal(), IdentityLink())

############## Prediction Error ######################################################################
preds = predict(linear_model_2, fitbit) #the total minutes of sleep our model predicts a person will get with corresponding total steps and total minutes active
resid = fitbit[!,5] - preds #residuals - shows how well our model is fitted to our dataset
histogram(resid, label = "residuals") #residuals are normally distributed
