import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv(r"D:\New\Downloads\Downloads\ML\train1.csv")

print("Data Loaded:")
print(dataset.head())



if "Id" in dataset.columns:
    dataset.drop(["Id"], axis=1, inplace=True)



dataset.replace("NA", pd.NA, inplace=True)

for col in dataset.columns:
    if pd.api.types.is_numeric_dtype(dataset[col]):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
    else:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])



dataset = pd.get_dummies(dataset)


X = dataset.drop("SalePrice", axis=1)
y = dataset["SalePrice"]


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
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel="rbf"),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}


# =====================
# Train + Evaluate
# =====================
print("\nMODEL RESULTS:\n")

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(name)
    print("R2 Score:", round(r2, 4))
    print("MSE:", round(mse, 2))
    print("----------------------")