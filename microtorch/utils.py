import numpy as np
from .engine import Tensor

#batch iterator
def batch_iterator(X,y,batch_size=32, shuffle=True):
    n = len(X)

    indices = np.arrange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0,n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_size],y[batch_idx]   


def accuracy_from_logits(logits,targets):
    preds = np.argmax(logits,axis = 1)
    return np.mean(preds ==targets)        




def evaluate_classifier(model, X, y, batch_size=64):
    total_correct = 0
    total = 0

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        x = Tensor(X_batch, requires_grad=False)
        logits = model(x)

        preds = np.argmax(logits.data, axis=1)
        total_correct += np.sum(preds == y_batch)
        total += len(y_batch)

    return total_correct / total
