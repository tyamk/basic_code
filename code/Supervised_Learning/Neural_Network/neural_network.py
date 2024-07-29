from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_digits()
X = data.images.reshape(len(data.images), -1)
print(len(data.images))
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(16,))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy =  accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy:.2f}')
