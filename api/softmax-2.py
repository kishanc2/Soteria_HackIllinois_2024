import pandas as pd 
import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
import pickle
import os

num_classes = 3
T = 1e2

# multiclass logistic regression
def softmax_train(train_data : np.ndarray, train_labels : np.ndarray, max_iter : int, lr : float) -> np.ndarray:
    N, D = train_data.shape
    W = np.random.rand(num_classes, D)
    for iter in range(max_iter):
        for i in range(N):
            x = train_data[i]
            y = train_labels[i]
            freq = np.exp((W @ x)/T)
            tot = np.sum(freq)
            for c in range(num_classes):
                if c != y:
                    W[c] -= lr * (np.exp((W[c] @ x)/T)/tot)*x
                else:
                    W[y] -= lr * (-x + (np.exp((W[c] @ x)/T)*x/tot))
    W /= N
    return W

def eval_softmax(W : np.ndarray, test_data : np.ndarray, test_labels : np.ndarray) -> float:
    N, D = test_data.shape
    predictions = np.array([np.argmax(np.exp((W @ test_data[i])/T)/np.sum(np.exp((W @ test_data[i])/T))) for i in range(N)])
    acc = np.sum([1 if test_labels[i] == predictions[i] else 0 for i in range(N)])/N 
    return acc

def main():
    # Read in Dataset
    df = pd.read_csv("cleaned.csv")
    # Map everything besides severity to integers (severity already in integers)
    ranked_df = df.iloc[:, :-1].rank(method='dense', ascending=False).astype(int)
    ranked_df["Accident_severity"] = df["Accident_severity"]
    stoi = {}
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            stoi[df.iloc[i, j]] = ranked_df.iloc[i, j]
    # X is datapoints with features, y is label set
    X, y = np.array(ranked_df.iloc[:,:-1]), np.array(ranked_df.iloc[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=20) 
    # Train model
    W = softmax_train(X_train, y_train, max_iter=100, lr=.0001)
    acc = eval_softmax(W, X_test, y_test)
    print(f"Accuracy: {acc}")
    with open("weights-2.pkl", "wb") as f:
        pickle.dump(W, f)
    f.close()
    with open('stoi.pkl', 'wb') as f:
        pickle.dump(stoi, f)
    f.close()

if __name__ == "__main__":
    main()