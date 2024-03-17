import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\dell\\Desktop\\PROG\\data mining\\ds_salaries.csv', encoding='ANSI')

df.head()

df.tail()

# Calculate the number of rows to delete (30% of the total rows)
rows_to_delete = int(0.3 * len(df))

# Randomly select rows to delete
rows_indices_to_delete = np.random.choice(df.index, size=rows_to_delete, replace=False)

# Mark the selected rows as missing or set to a specific value
df.loc[rows_indices_to_delete, 'salary'] = np.nan  # Replace 'Column1' with the column you want to modify
df.loc[rows_indices_to_delete, 'salary_in_usd'] = np.nan # Replace 'Column2' with the column you want to modify

# Optional: Reset the index if needed
df.reset_index(drop=True, inplace=True)

#drop duplicate rows
duplicate_rows = df[df.duplicated()]
df_no_duplicates = df.drop_duplicates()

#droping missing values
df.dropna(inplace=True)
df_cleaned = df.dropna()

#Splitting Numerical from Categorical features
numerical_features = list(set(df.columns.to_list()) - {'salary','salary_in_usd','remote_ratio','work_year'})
categorical_features = list(set(df.columns.to_list()) - set(numerical_features))

#select which column to normalize
column_to_normalize = 'salary_in_usd'
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Normalize the selected column
df[column_to_normalize] = scaler.fit_transform(df[[column_to_normalize]])

#discretization of salary_in_usd
column_to_discretize = 'salary_in_usd'

# Define the bin edges (intervals) for discretization
bin_edges = [0.000622657,0.306592,0.914581,1]

# Define labels for the bins
bin_labels = ['LOW', 'MEDIUM', 'HIGH']

# Use the cut function to discretize the selected column
df['discretized_' + column_to_discretize] = pd.cut(df[column_to_discretize], bins=bin_edges, labels=bin_labels)

# Label encode the 'discretized_salary_in_usd' column
le = LabelEncoder()
df['discretized_salary_in_usd'] = le.fit_transform(df['discretized_salary_in_usd'])

#desribe the median of salary_in_usd attribute
attribute_column = 'salary_in_usd'
median_value = df[attribute_column].median()
#print the median value of salary_in_usd
print(f"The median value of the '{attribute_column}' column is: {median_value}")

# Filtering and keeping only rows where 'salary_in_usd' is greater than 0
df = df[df['salary_in_usd'] > 0]

#filtering and keeping only rows where 'discretized_salary_in_usd' is not nan
df = df[df['discretized_salary_in_usd'] != np.nan]

# Print the modified dataset
print(df)
print(df.isnull().sum())
print(df.describe())

# Select the column for analysis
column_of_interest = 'discretized_salary_in_usd'

# Generate a histogram
plt.figure(figsize=(8, 6))
plt.hist(df[column_of_interest], bins=10, color='blue', edgecolor='black')
plt.title(f'Histogram of {column_of_interest}')
plt.xlabel(column_of_interest)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Generate a boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_of_interest], vert=False)
plt.title(f'Boxplot of {column_of_interest}')
plt.xlabel(column_of_interest)

# Calculate the IQR (Interquartile Range)
Q1 = df[column_of_interest].quantile(0.25)
Q3 = df[column_of_interest].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify and remove outliers
outliers = (df[column_of_interest] < lower_bound) | (df[column_of_interest] > upper_bound)
df_no_outliers = df[~outliers]

# Create a boxplot of the data without outliers
plt.boxplot(df_no_outliers[column_of_interest], vert=False, positions=[2])  # Use positions to avoid overlapping with the original boxplot
plt.show()

#create countplots for numerical attributes
sns.countplot(x=df['salary'])
plt.show()
sns.countplot(x=df['salary_in_usd'])
plt.show()
sns.countplot(x=df['remote_ratio'])
plt.show()
sns.countplot(x=df['work_year'])
plt.show()

# Assuming df_salaries is our DataFrame
# Assuming 'company_size' is one of our features and 'salary_in_usd' is the target variable

le = LabelEncoder()

 # create a copy of the dataframe
df_copy = df.copy()
df_copy['company_size'] = le.fit_transform(df_copy['company_size'])
print(le.classes_)

 # Drop columns that are not needed for regression
df_copy.drop(columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_currency',
                                'employee_residence', 'company_location'], inplace=True)

 # Assuming 'company_size' is a feature and 'salary_in_usd' is the target variable
X = df_copy[['company_size']]
y = df_copy['salary_in_usd']

 # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Fit Bayesian Ridge Regression
regressor = BayesianRidge()
regressor.fit(X_train, y_train)

 # Make predictions on the test set
y_pred = regressor.predict(X_test)

 # Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

 # Visualize the regression line
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.title('Bayesian Ridge Regression')
plt.xlabel('Company Size (Encoded)')
plt.ylabel('Salary in USD')
plt.legend()
plt.show()
 #////////////////////////////////
 # Create a copy of the dataframe
df_copy = df.copy()

 # Drop columns that are not needed for classification
df_copy.drop(columns=['work_year', 'experience_level', 'salary_in_usd', 'salary_currency',
                                'employee_residence', 'company_location', 'company_size'], inplace=True)

 # Assuming 'job_title' is a feature and 'employment_type' is the target variable
X = df_copy['job_title']
y = df_copy['employment_type']

 # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Use CountVectorizer to convert job titles into numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

 # Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, y_train)

 # Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_vectorized)

 # Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
cm = confusion_matrix(y_test, y_pred)

 # Create a heatmap for the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=naive_bayes_classifier.classes_, yticklabels=naive_bayes_classifier.classes_)
plt.title('Confusion Matrix "naive bayes"')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
 #////////////////////////////////

job_count = df['job_title'].value_counts().nlargest(10)

 # Calculate percentages
total_jobs = len(df['job_title'])
percentages = (job_count / total_jobs) * 100

 # Create a horizontal bar plot
ax = sns.barplot(x=job_count.values, y=job_count.index, orient='h')

 # Annotate bars with percentages
for i, v in enumerate(job_count.values):
     percentage = percentages.iloc[i]
     ax.text(v + 1, i, f'{percentage:.2f}%', va='center', fontsize=10)

 # Customize plot labels and title
plt.title('the Job Titles in Dataset with Percentages')
plt.xlabel('Frequency')
plt.ylabel('Job Title the 10')
 # Display the plot
plt.show()


 # Create a count plot for experience levels
ax = sns.countplot(x=df['experience_level'])

 # Annotate bars with counts or percentages
total_records = len(df['experience_level'])
counts = df['experience_level'].value_counts()

 # Calculate percentages
percentages = (counts / total_records) * 100

 # Annotate bars with percentages or counts
for i, v in enumerate(counts):
     percentage = percentages.iloc[i]
     ax.text(i, v + 1, f'{v} ({percentage:.2f}%)', ha='center', va='bottom', fontsize=10)

 # Customize plot labels and title
plt.title('Experience Levels Distribution')
plt.xlabel('Experience Level')
plt.ylabel('Count')
plt.show()



le = LabelEncoder()

 # create a copy of the dataframe
df_copy = df.copy()
df_copy['company_size'] = le.fit_transform(df_copy['company_size'])
print(le.classes_)

 # Drop columns that are not needed for clustering
df_copy.drop(columns=['work_year', 'experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'salary_currency',
                               'employee_residence', 'company_location'], inplace=True)

silhouette_scores = []
for k in range(2, 10):  # Start from 2 clusters as silhouette score requires at least 2 clusters
     kmeans = KMeans(n_clusters=k)
     kmeans.fit(df_copy)
     score = silhouette_score(df_copy, kmeans.labels_)
     silhouette_scores.append(score)

plt.plot(range(2, 10), silhouette_scores)
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(df_copy)

df['clusters'] = kmeans.predict(df_copy)
sns.scatterplot(df, x='work_year', y='salary', hue='clusters')
plt.gcf().set_size_inches(8,7)
plt.tight_layout()
plt.show();
 #//////////////////////////////////////////
