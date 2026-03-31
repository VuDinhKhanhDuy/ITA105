import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [2, 4, 5, 6, 7, 8, 8.5, 9]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)


new_hours = pd.DataFrame([[6]], columns=["Hours"])
predicted_score = model.predict(new_hours)
print(predicted_score)


new_data = pd.DataFrame([[4], [6], [9]], columns=["Hours"])
predictions = model.predict(new_data)
print(predictions)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.show()

y_pred = model.predict(X)
score = r2_score(y, y_pred)
print("R2 Score:", score)