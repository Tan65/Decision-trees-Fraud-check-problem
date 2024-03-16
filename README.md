# Decision-trees-Fraud-check-problem
Use decision trees to prepare a model on fraud data  treating those who have taxable_income &lt;= 30000 as "Risky" and others are "Good"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Fraud_check.csv')

# Treat taxable_income <= 30000 as "Risky" and others as "Good"
data['Taxable.Income'] = data['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Undergrad', 'Marital.Status', 'Urban'])

# Univariate Analysis: Histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i+1)  # Adjusted to 3 rows and 3 columns
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Bivariate Analysis: Pairplot
sns.pairplot(data, hue='Taxable.Income')
plt.suptitle('Pairplot of Variables')
plt.show()

# Multivariate Analysis: Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Train the decision tree model
X = data.drop('Taxable.Income', axis=1)
y = data['Taxable.Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
