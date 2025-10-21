# ----------------------------
# Import Required Libraries
# ----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ----------------------------
# Task 1: Load and Explore the Dataset
# ----------------------------

try:
    # Load Iris dataset from sklearn
    iris_data = load_iris()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset successfully loaded!\n")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the dataset path.")
except Exception as e:
    print("❌ An error occurred while loading the dataset:", e)

# Display first few rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Check dataset structure and missing values
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (no missing values in Iris dataset, but showing how to handle)
df = df.dropna()
print("\n✅ Dataset cleaned (if any missing values existed).")

# ----------------------------
# Task 2: Basic Data Analysis
# ----------------------------

# Descriptive statistics
print("\n📊 Basic Statistics:")
print(df.describe())

# Group by species and compute mean
species_means = df.groupby('species').mean(numeric_only=True)
print("\nAverage values per species:")
print(species_means)

# Observations
print("\n🔍 Observations:")
print("- Iris-setosa generally has smaller petal and sepal measurements.")
print("- Iris-virginica has the largest average petal length and width.")
print("- Iris-versicolor falls between the two other species.")

# ----------------------------
# Task 3: Data Visualization
# ----------------------------

sns.set(style="whitegrid")

# 1️⃣ Line Chart - Petal Length trend (using index as time-like axis)
plt.figure(figsize=(8,5))
plt.plot(df.index, df['petal length (cm)'], label='Petal Length', color='green')
plt.title('Line Chart: Petal Length Trend')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# 2️⃣ Bar Chart - Average Petal Length per Species
plt.figure(figsize=(7,5))
species_means['petal length (cm)'].plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Bar Chart: Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3️⃣ Histogram - Distribution of Sepal Length
plt.figure(figsize=(7,5))
plt.hist(df['sepal length (cm)'], bins=15, color='purple', edgecolor='black')
plt.title('Histogram: Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4️⃣ Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2')
plt.title('Scatter Plot: Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# ----------------------------
# Task 4: Findings and Observations
# ----------------------------

print("\n📈 Findings Summary:")
print("1. Sepal and petal lengths show strong positive correlation — longer sepals tend to have longer petals.")
print("2. Each species has distinct petal dimensions, useful for classification.")
print("3. Petal length trends vary across the dataset, but remain species-consistent.")
print("4. The dataset is clean, balanced, and ideal for supervised learning tasks like classification.")
