import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

%matplotlib inline

matches = pd.read_csv('matches.csv')
matches.head()
# Summary statistics for winner and loser ages
age_stats = matches[['winner_age', 'loser_age']].describe()
print(age_stats)
print(matches['surface'].unique())

# Histograms for winner and loser ages, color-coded by surface
plt.figure(figsize=(12, 8))
sns.histplot(data=matches, x='winner_age', hue='surface', bins=20, kde=True, multiple='stack')
sns.histplot(data=matches, x='loser_age', hue='surface', bins=20, kde=True, multiple='stack')
plt.title('Distribution of Winner and Loser Ages by Surface Type')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(title='Surface')
plt.show()

# Read data from the CSV file
#matches = pd.read_csv('matches.csv')

# Ensure that 'surface' values are consistent and correct in your dataset
matches['surface'] = matches['surface'].replace({'Clay': 'clay', 'Grass': 'grass', 'Carpet': 'carpet', 'Hard': 'hard'})

# Define a custom color palette for each surface
surface_palette = {'clay': 'green', 'grass': 'blue', 'carpet': 'purple', 'hard': 'red'}

# Histogram for winner ages, color-coded by surface
plt.figure(figsize=(12, 8))

sns.histplot(data=matches, x='winner_age', hue='surface', bins=20, kde=True, multiple='stack', palette=surface_palette)

plt.title('Distribution of Winner Ages by Surface Type')
plt.xlabel('Winner Age')
plt.ylabel('Frequency')

# Customize legend labels
plt.legend(title='Surface', labels=['Clay', 'Grass', 'Carpet', 'Hard'])

plt.show()

# Read data from the CSV file
#matches = pd.read_csv('matches.csv')

# Ensure that 'surface' values are consistent and correct in your dataset
matches['surface'] = matches['surface'].replace({'Clay': 'clay', 'Grass': 'grass', 'Carpet': 'carpet', 'Hard': 'hard'})

# Define a custom color palette for each surface
surface_palette = {'clay': 'green', 'grass': 'blue', 'carpet': 'purple', 'hard': 'red'}

# Histogram for loser ages, color-coded by surface
plt.figure(figsize=(12, 8))

sns.histplot(data=matches, x='loser_age', hue='surface', bins=20, kde=True, multiple='stack', palette=surface_palette)

plt.title('Distribution of Loser Ages by Surface Type')
plt.xlabel('Loser Age')
plt.ylabel('Frequency')

# Customize legend labels
plt.legend(title='Surface', labels=['Clay', 'Grass', 'Carpet', 'Hard'])

plt.show()

plt.figure(figsize=(12, 8))
sns.jointplot(x=matches['winner_age'], y=matches['loser_age'], kind='hex', cmap='viridis')
plt.suptitle('Hexbin Plot of Winner Age vs. Loser Age', y=1.02)
plt.show()

# Heatmap for winner and loser ages by surface
age_surface_pivot = matches.pivot_table(index='winner_age', columns='surface', aggfunc='size', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(age_surface_pivot, cmap='YlGnBu', cbar_kws={'label': 'Frequency'})
plt.title('Heatmap of Winner Ages by Surface Type')
plt.xlabel('Surface')
plt.ylabel('Winner Age')
plt.show()

# Read data from the CSV file
#df = pd.read_csv('matches.csv')

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Map surface types to categorical codes
surface_codes = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
matches['surface_code'] = matches['surface'].map(surface_codes)

# Map surface codes to colors for better visualization
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
matches['color'] = matches['surface_code'].map(colors)

# Scatter plot
ax.scatter(matches['winner_age'], matches['loser_age'], matches['surface_code'], c=matches['color'], s=50)

# Set labels
ax.set_xlabel('Winner Age')
ax.set_ylabel('Loser Age')
ax.set_zlabel('Surface')

# Set legend for surface colors
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=surface,
                             markerfacecolor=colors[code], markersize=10) for surface, code in surface_codes.items()]
ax.legend(handles=legend_labels, title='Surface')

# Show the plot
plt.show()


# Define features and target variable
features = ['winner_age', 'loser_age', 'surface']
target = 'winner_rank'

# Handling missing values in the target variable
matches[target].fillna(matches[target].mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(matches[features], matches[target], test_size=0.2, random_state=42)

# Preprocessing: Standardize numerical features and one-hot encode categorical features
numeric_features = ['winner_age', 'loser_age']
categorical_features = ['surface']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Build the pipeline with linear regression as the regressor
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', LinearRegression())])

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions on the test set
y_pred = pipeline.predict(X_test)

matches.winner_rank.describe()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
