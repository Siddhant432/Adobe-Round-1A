# train_model.py — Enhanced for Adobe Hackathon Round 1A
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Features: [font_size, is_bold, word_count, char_count]
X = np.array([
    # H1 (large, bold, short)
    [24, 1, 2, 15], [23, 1, 3, 20], [25, 1, 4, 25],

    # H2 (medium-large, bold)
    [20, 1, 3, 25], [19, 1, 4, 30], [20, 1, 5, 35],

    # H3 (medium, bold)
    [16, 1, 5, 40], [15, 1, 6, 42], [15, 1, 6, 45],

    # Body text (not bold, long)
    [12, 0, 20, 120], [11, 0, 25, 160], [10, 0, 30, 180],
    [13, 0, 18, 100], [12, 0, 22, 140],

    # Filler text / copyright / invalid heading
    [11, 0, 3, 60], [10, 0, 2, 45], [9, 0, 5, 80],
    [10, 0, 1, 30], [12, 0, 2, 50]
])

# Labels: 3 = H1, 2 = H2, 1 = H3, 0 = None/body
Y = np.array([
    3, 3, 3,  # H1
    2, 2, 2,  # H2
    1, 1, 1,  # H3
    0, 0, 0, 0, 0,  # paragraph
    0, 0, 0, 0, 0   # filler
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
clf.fit(X_scaled, Y)

joblib.dump(clf, "heading_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Trained model and scaler saved.")
