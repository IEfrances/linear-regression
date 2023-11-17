#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = np.dot(X, w)
    loss = np.mean((y_hat - y) ** 2) / 2
    risk = np.mean(np.abs(y_hat - y))

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val,hyperparameters):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    MaxIter, alpha, batch_size, decay, _, _ = hyperparameters


    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):
            
            start = int(b * batch_size)
            end = min(int((b + 1) * batch_size), N_train) 

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descentc
            gradient = np.dot(X_batch.T, (y_hat_batch - y_batch)) / len(y_batch)
            w = w - alpha * gradient

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        total_batches = (N_train / batch_size)
        loss_per_batch = loss_this_epoch / total_batches
        losses_train.append(loss_per_batch)

        # 2. Perform validation on the validation set by the risk
        y_val_hat, _, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            risk_best = risk_val
            epoch_best = epoch
            w_best = w.copy()

    _, risk_test, _ = predict(X_test, w_best, y_test)
    # Return some variables as needed
    return epoch_best, risk_best, risk_test, losses_train, risks_val


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable

# Extend features
X_extend = np.concatenate((X_[:, :1], X_[:, 1:]**2, X_[:, 1:2] * X_[:, 2:3], X_[:, 2:]**2), axis=1)


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_extend)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_extend[:300]
y_train = y[:300]

X_val = X_extend[300:400]
y_val = y[300:400]

X_test = X_extend[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay

hyperparameters = [MaxIter, alpha, batch_size, decay, 0, [3, 1, 0.3, 0.1, 0.03, 0.01]]

best_hyperparameter = None
best_val_performance = float('inf')
best_epoch = 0
best_losses = []
best_risks = []

for hyperparameter in hyperparameters[5]:
    hyperparams = [MaxIter, alpha, batch_size, decay, 0, hyperparameter]
    epoch, val_performance, test_performance, losses, risks = train(X_train, y_train, X_val, y_val, hyperparams)
    if val_performance < best_val_performance:
        best_hyperparameter = hyperparameter
        best_val_performance = val_performance
        best_epoch = epoch
        best_losses = losses
        best_risks = risks

print("Number of epoch that yields the best validation performance:", best_epoch)
print("Validation performance (risk) in that epoch:", best_val_performance)
print("Test performance (risk) in that epoch:", test_performance)
print("Best hyperparameter:", best_hyperparameter)

# Plot the learning curve for training loss
plt.figure(figsize=(12, 6))
plt.plot(range(MaxIter), best_losses, label='Training Loss', color='green')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('Learning Curve of The Training Loss')
plt.legend()
plt.savefig('training_loss_curve.jpg')

# Plot the learning curve for validation risk
plt.figure(figsize=(12, 6))
plt.plot(range(MaxIter), best_risks, label='Validation Risk', color='red')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Risk')
plt.title('Learning Curve of The Validation Risk')
plt.legend()
plt.savefig('validation_risk_curve.jpg')