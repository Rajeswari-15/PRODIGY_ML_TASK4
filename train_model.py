import joblib
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from preprocess_data import load_data
from sklearn.metrics import accuracy_score

def train():
    print("Loading data...")
    X_train, X_test, Y_train, Y_test = load_data()
    print(f"Original data shape: {X_train.shape}")

    print("Applying PCA...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Reduced data shape: {X_train_pca.shape}")

    print("Training SVM...")
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_pca, Y_train)

    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(pca, "pca_model.pkl")
    joblib.dump(model, "gesture_model.pkl")
    print("Saved PCA and model.")

if __name__ == "__main__":
    train()
