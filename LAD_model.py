from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


LAD_recordings = np.load('../data/LAD_train.npy')
print(len(LAD_recordings))
y = np.load('../data/labels_500.npy')
X_train, X_test, y_train, y_test = train_test_split(LAD_recordings, y, test_size=0.3, random_state=0)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X_train, y_train)
clf.fit(LAD_recordings, y)

# y_pred = clf.predict(X_test)
y_pred = clf.predict(LAD_recordings)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
acc = (tn + tp) / (fp + fn + tn + tp)
print(acc)

scores = cross_val_score(clf, LAD_recordings, y, cv=5)

print(scores)
