import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


dataset = pd.read_csv(
    r"D:\New\Downloads\Downloads\ML\train.csv"
)

print(dataset.head())



dataset.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1,
    inplace=True
)



dataset["Age"] = dataset["Age"].fillna(
    dataset["Age"].mean()
)

dataset["Fare"] = dataset["Fare"].fillna(
    dataset["Fare"].mean()
)

dataset["Embarked"] = dataset["Embarked"].fillna(
    dataset["Embarked"].mode()[0]
)


le = LabelEncoder()

# male/female → 0/1
dataset["Sex"] = le.fit_transform(
    dataset["Sex"]
)

# S,C,Q → numbers
dataset["Embarked"] = le.fit_transform(
    dataset["Embarked"]
)


print("\nAfter Encoding:")
print(dataset.head())


X = dataset.drop("Survived", axis=1)
y = dataset["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression":
        LogisticRegression(),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(kernel="linear"),

    "Kernel SVM":
        SVC(kernel="rbf"),

    "Naive Bayes":
        GaussianNB(),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier()
}


print("\nResults:\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc*100:.2f}%")