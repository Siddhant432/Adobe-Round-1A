# train_model.py
# Simulates training the heading classifier using fake data (for demo purpose)

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dummy data: [font_size, is_bold, num_words, num_chars]
X = np.array([
    [24, 1, 3, 20],  # Title
    [18, 1, 4, 25],  # H1
    [16, 1, 5, 30],  # H2
    [14, 1, 6, 35],  # H3
    [12, 0, 15, 80], # Body
    [10, 0, 20, 120], # Body
])

# Labels: 3=H1, 2=H2, 1=H3, 0=None
# Title is ignored from training, inferred separately
Y = np.array([3, 2, 1, 1, 0, 0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "heading_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
