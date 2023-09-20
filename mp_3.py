import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10000, noise=0.4, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

dtc_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
dtc_entropy.fit(X_train, y_train)
y_pred_entropy = dtc_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

dtc_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
dtc_gini.fit(X_train, y_train)
y_pred_gini = dtc_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)

rfc = RandomForestClassifier(n_estimators=100, random_state=1)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

voting_clf = VotingClassifier(
    estimators=[('dtc_entropy', dtc_entropy), ('dtc_gini', dtc_gini), ('rfc', rfc), ('lr', lr), ('svm', svm)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)

print("Classifier Insights Summary:")
print("\nDecision Tree Classifier (Entropy):")
print("  - Accuracy: {:.4f}".format(accuracy_entropy))
print("\nDecision Tree Classifier (Gini):")
print("  - Accuracy: {:.4f}".format(accuracy_gini))
print("\nRandom Forest Classifier:")
print("  - Accuracy: {:.4f}".format(accuracy_rfc))
print("\nLogistic Regression Classifier:")
print("  - Accuracy: {:.4f}".format(accuracy_lr))
print("\nSVM Classifier:")
print("  - Accuracy: {:.4f}".format(accuracy_svm))
print("\nVoting Classifier (Ensemble):")
print("  - Accuracy: {:.4f}".format(accuracy_voting))

accuracies = {
    "Decision Tree (Entropy)": accuracy_entropy,
    "Decision Tree (Gini)": accuracy_gini,
    "Random Forest": accuracy_rfc,
    "Logistic Regression": accuracy_lr,
    "SVM": accuracy_svm,
    "Voting Classifier": accuracy_voting,
}

plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Classifier Comparison")
plt.ylim(0.8, 0.9)
plt.xticks(rotation=45)
plt.show()
