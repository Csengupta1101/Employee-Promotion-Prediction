# Employee-Promotion-Prediction

# Understanding the Problem Statement Given To Us.
● HR analytics is revolutionising the way human resources departments operate,
leading to higher efficiency and better results overall for the Organization
● The collection, processing and analysis of data is manual, and nature of HR
dynamics and KPIs has been constraining HR Dept.
● Try this predictive analytics and identify the employees who are most likely to get
promoted in an Organization

# The Client Requirement

It is a Large MNC and they have 9 broad verticals across the organisation. One of the
problems they face is around identifying the right people for promotion and preparing
them in time.
Currently there Process involves:
● Identify employees based on Past performance and Recommendation.
● Selected Employees go through Training and Evaluation based on Skills Required.
● At the end, The employee gets Promotion based on Training Score, and KPIs.

# Necessary Libraries¶
We Import Numpy, Pandas, Matplot, and Seaborn for Data Analysis and Visualizations
We import ipywidgets, Sweetviz, ppscore for Exploratory Data Analysis
We Import Sklearn, Imblearn for Machine Learning Modelling


# Outlier treatment 

It is quite clear that we are not having Outliers in our Dataset, the average training score for most of the Employee lie between 40 to 100, 
which is a very good distribution, also th mean is 50.Also, the Length of service, is not having very disruptive values, so we can keep them for model training. 
they are not going to harm us a lot.

Here, using the Box plot, helps us to analyze the middle 50 percentile of the data, and we can clearly check the minimum, maximum, median, and outlier values.
In the Length of service attribute, we can see some points after the Max Value, which can be termed to be as Outliers. 
We do not need to remove these values, as the values are not very far and Huge.

# Missing Value treatment

we can easily, see that the Target Class is Highly Imbalanced, and we must balance these classes of Target Class. 
Most of the Times, when we use Machine Learning Models with Imbalanced Classes, we have very poor Results which are completely biased 
towards the class having Higher Distribution.

we imputed the missing values, using the Mode values, even for the previous year rating, it only seems to be numerical, but in real it's also categorical.
After, Imputing the missing values in the training and testing data set we can see that there are no Null Values left in any of the datasets.
So, we are Done with the Treatment of the Missing Values.

We can see from the above table, that Only two columns have missing values in Train and Test Dataset both. 
Also, the Percentage of Missing values is around 4 and 7% in education, and previous_year_rating respectively. 
So, do not have delete any missing values, we can simply impute the values using Mean, Median, and Mode Values. 

# Univariate analysis

We, can see after plotting  some pie charts, we have for representing KPIs, Previous year Ratings, and Awards Won?
Also, The one Big Pattern is that only some of the employees could reach above 80% of KPIs set.
Most of the Employees have a very low rating for the previous year, and
very few employees, probably 2% of them could get awards for their work, which is normal.

From, the above pie charts displayed for representing Education, Gender, and Recruitment Channel.

lets infer the Main Highlights

Very Few employees are actually working only after their Secondary Education, 
Obviously Females are again in Minority as compared to their Male Counterparts.
and the Recruitment Channel, says that the Referred Employees are very less, i.e., 
most of the employees are recruited either by sourcing, or some other recruitment agencies, sources etc.

# Bivariate analysis

As we have already seen that the Females are in Minority, but when it comes to Promotion, they are competing with their Men Counterparts neck-to-neck. 
That's a great Inference.

From, the above chart we can see that almost all the Departments have a very similar effect on Promotion. 
So, we can consider that all the Departments have a similar effect on the promotion. 
Also, this column comes out to be lesser important in making a Machine Learning Model,
as it does not contribute at all when it comes to Predicting whether the Employee should get Promotion.

# Multivariate analysis

Here, we can see some obvious results, that is Length of Service, and Age are Highly Correlated,
Also, KPIs, and Previous year rating are correlated to some extent, hinting that there is some relation.

# Feature Engineering
Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. Feature engineering can be considered as applied machine learning itself.

There are mutliple ways of performing feature engineering.

So many people in the Industry consider it the most important step to improve the Model Performance.

We should always understand the columns well to make some new features using the old existing features.

Let's discuss the ways how we can perform feature engineering

We can perform Feature Engineering by Removing Unnecassary Columns
We can do it by Extracting Features from the Date and Time Features.
We can do it by Extracting Features from the Categorcial Features.
We can do it by Binnning the Numerical and Categorical Features.
We can do it by Aggregating Multiple Features together by using simple Arithmetic operations
Here, we are only going to perform Feature Engineering by Aggregating some features together

# Dealing with Categorical Columns¶

Categorical variables are known to hide and mask lots of interesting information in a data set. It’s crucial to learn the methods of dealing with such variables. If you won’t, many a times, you’d miss out on finding the most important variables in a model. It has happened with me. Initially, I used to focus more on numerical variables. Hence, never actually got an accurate model. But, later I discovered my flaws and learnt the art of dealing with such variables.

There are various ways to encode categorical columns into Numerical columns
This is an Essential Step, as we Machine Learning Models only works with Numerical Values.
Here, we are going to use Business Logic to encode the education column
Then we will use the Label Encoder, to Department and Gender Columns

# Splitting the Data

This is one of the most Important step to perform Machine Learning Prediction on a Dataset, We have to separate the Target and Independent Columns.

We store the Target Variable in y, and then we store the rest of the columns in x, by deleting the target column from the data
Also, we are changing the name of test dataset to x_test for ease of understanding.

# Resampling¶
Resampling is the method that consists of drawing repeated samples from the original data samples. The method of Resampling is a nonparametric method of statistical inference.

Earlier, in this Problem we noticed that the Target column is Highly Imbalanced, we need to balance the data by using some Statistical Methods.
There are many Statistical Methods we can use for Resampling the Data such as:
Over Samping
Cluster based Sampling
Under Sampling.
Oversampling and undersampling in data analysis are techniques used to adjust the class distribution of a data set. These terms are used both in statistical sampling, survey design methodology and in machine learning. Oversampling and undersampling are opposite and roughly equivalent techniques

We are going to use Over Sampling.
We will not use Under Sampling to avoid data loss.

# Feature Scaling

Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, 
it is also known as data normalization and is generally performed during the data preprocessing step.

# Machine Learning Predictive Modelling

Predictive modeling is a process that uses data and statistics to predict outcomes with data models.
These models can be used to predict anything from sports outcomes and TV ratings to technological advances and corporate earnings. 
Predictive modeling is also often referred to as: Predictive analytics.

# Decision Tree Classifier

A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, 
including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

# Findings

> department -> The values are from 0 to 8, (Department does not matter a lot for promotion)
> education -> The values are from 0 to 3 where Masters-> 3, Btech -> 2, and secondary ed -> 1
> gender -> the values are 0 for female, and 1 for male
> no_of_trainings -> the values are from 0 to 5
> age -> the values are from 20 to 60
> previou_year_rating -> The values are from 1 to 5
> length_of service -> The values are from 1 to 37
> KPIs_met >80% -> 0 for Not Met and 1 for Met
> awards_won> -> 0-no, and 1-yes
> avg_training_score -> ranges from 40 to 99
> sum_metric -> ranges from 1 to 7
> total_score -> 40 to 710

# Conclusion

Hope you've liked the submisson and detailing. Would love to hear your thoughts. Please comment with any suggested input.


