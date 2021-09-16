import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score


diabetes = pd.read_csv('diabetes.csv')
df = diabetes.copy()
target = 'Outcome'

# Separating X and y
X = df.drop('Outcome', axis = 1)
Y = df['Outcome']

chi2_check = []
for column in X.columns:
    if (chi2_contingency(pd.crosstab(df['Outcome'], df[column]))[1] < 0.05):
        chi2_check.append('Reject Null Hypothesis')
    else:
        chi2_check.append('Fail to Reject Null Hypothesis')
res = pd.DataFrame(data = [X.columns, chi2_check]).T 
res.columns = ['Column', 'Hypothesis']

X_selected = X[res[res['Hypothesis'] == 'Reject Null Hypothesis']['Column']]
X_selected

X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size = 0.3, random_state = 0)

# Build decision tree model
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X, Y) 

# Saving the model
import pickle
pickle.dump(dtc, open('diabetes_dtc.pkl', 'wb'))