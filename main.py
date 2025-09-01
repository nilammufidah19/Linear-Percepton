import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearPerceptron:
    def __init__(self, input_dim, learning_rate=0.1, n_iter=1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = np.zeros(input_dim)
        self.bias = 0

        # logging
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss_accuracy(self, X, y):
        y_pred = self.predict_proba(X)
        loss = -np.mean(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
        y_class = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_class == y)
        return loss, accuracy

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m = len(y_train)
        for epoch in range(1, self.n_iter + 1):
            # forward pass training
            y_pred = self.predict_proba(X_train)

            # gradient descent update
            dw = (1/m) * np.dot(X_train.T, (y_pred - y_train))
            db = (1/m) * np.sum(y_pred - y_train)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # training metrics
            train_loss, train_acc = self.compute_loss_accuracy(X_train, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # validation metrics
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.compute_loss_accuracy(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                msg = (f"Epoch {epoch:4d} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            else:
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)
                msg = (f"Epoch {epoch:4d} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            if epoch % 1 == 0 or epoch == 1 or epoch == self.n_iter:
                print(msg)

    def plot_history(self):
        epochs = range(1, self.n_iter + 1)

        # Loss plot
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        if any(v is not None for v in self.history["val_loss"]):
            plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()

        # Accuracy plot
        plt.subplot(1,2,2)
        plt.plot(epochs, self.history["train_acc"], label="Train Acc")
        if any(v is not None for v in self.history["val_acc"]):
            plt.plot(epochs, self.history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Epoch")
        plt.legend()

        plt.show()

if __name__ == '__main__':
    df = pd.read_csv("split_data.csv")
    print(df.tail())

    data_train = df[df["type_data"]=="train"]
    data_val = df[df["type_data"]=="test"]

    y_train = data_train["TARGET"].to_numpy()
    y_val = data_val["TARGET"].to_numpy()

    # x_train = data_train.drop(columns=["TARGET"]).to_numpy()
    # x_val = data_val.drop(columns=["TARGET"]).to_numpy()

    x_train = data_train[["X1", "X2", "X3", "X4"]].to_numpy()
    x_val = data_val[["X1", "X2", "X3", "X4"]].to_numpy()

    model = LinearPerceptron(input_dim=4, learning_rate=0.1, n_iter=100)
    model.fit(x_train, y_train, x_val, y_val)

    print("\nBobot:", model.weights)
    print("Bias:", model.bias)

    # Plot history
    model.plot_history()