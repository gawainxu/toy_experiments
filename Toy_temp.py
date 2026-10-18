import argparse
import pickle
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def getArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path_train", type=str, default="./features2/toy_toy_E5_task_1_data_1_train_conditional")
    parser.add_argument("--feature_path_test", type=str, default="./features2/toy_toy_E5_task_1_data_1_test_conditional")
    parser.add_argument("--remove_third_class", type=int, default=0)
    parser.add_argument("--layer_name", type=str, default="linear2")

    opt = parser.parse_args()
    return opt



def regression(train_features, train_labels, test_features, test_labels):

    clf = LogisticRegression(
    C=1.,
    max_iter=5000,
    class_weight="balanced",
    random_state=0,)

    clf.fit(train_features, train_labels)
    accuracy = clf.score(test_features, test_labels)
    print("accuracy:", accuracy)

    return accuracy


class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        # Output dimension is 1 for binary classification
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Returns raw logits (numerically stable for BCEWithLogitsLoss)
        return self.linear(x)



if __name__ == "__main__":
    opt = getArgs()

    with open(opt.feature_path_train, "rb") as f:
        features_train, labels_train = pickle.load(f)

    with open(opt.feature_path_test, "rb") as f:
        features_test, labels_test = pickle.load(f)
    
    features_train = [np.squeeze(f[opt.layer_name].numpy()) for f in features_train]
    features_test = [np.squeeze(f[opt.layer_name].numpy()) for f in features_test]
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    #labels_train = [i-min(labels_train) for i in labels_train]
    #labels_test = [i-min(labels_test) for i in labels_test]

    if opt.remove_third_class == 1:
        third_class_train = [i for i, l in enumerate(labels_train) if l < 2]
        third_class_test = [i for i, l in enumerate(labels_test) if l < 2]
        features_train = features_train[third_class_train]
        labels_train = [labels_train[i] for i in third_class_train]
        features_test = features_test[third_class_test]
        labels_test = [labels_test[i] for i in third_class_test]

    #accuracy = regression(features_train, labels_train, features_test, labels_test)
    #accuracy = regression(features_train, labels_train, features_train, labels_train)

    # Instantiate model
    model = LogisticRegressionPyTorch(input_dim=20)
    criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid + Binary Cross Entropy

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    train_dataset = TensorDataset(
        torch.tensor(features_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)
    )
    test_dataset = TensorDataset(
        torch.tensor(features_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    C_param = 1.0
    weight_decay = 1.0 / C_param
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            #X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")

    # 5. Evaluation & Inference
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            #X_batch = X_batch.cuda()

            logits = model(X_batch)
            probs = torch.sigmoid(logits)  # Probabilities equivalent to predict_proba
            preds = (probs >= 0.5).float()  # Hard labels equivalent to predict

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    auroc = roc_auc_score(all_targets, all_probs)

    print(f"\n--- Evaluation Results ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUROC:    {auroc:.4f}")