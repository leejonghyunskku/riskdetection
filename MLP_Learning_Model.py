import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb
import glob
import os
from sklearn.utils.class_weight import compute_class_weight
import numexpr
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


data_dir = "/home/work/dataset"  
csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

print(f"발견된 CSV 파일 개수: {len(csv_files)}")
for f in csv_files:
    print(f)

df_list = []
for f in csv_files:
    tmp = pd.read_csv(f)
    tmp["source_file"] = os.path.basename(f)
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)
df1 = df.copy()
df2 = df.copy()

print(f"DataFrame shape: {df.shape}")

# 라벨 매핑
label_to_id = {'normal': 0, 'assault': 1, 'robbery': 2, 'swoon': 3}

num_classes = len(label_to_id)

def map_labels(cell):
    onehot = np.zeros(num_classes, dtype=int)
    if pd.isna(cell):
        return onehot
    labels = [label_to_id[label.strip()] for label in str(cell).split(",") if label.strip() in label_to_id]
    for l in labels:
        onehot[l] = 1
    return onehot

def map_labels_custom(cell):
    vec = np.zeros(4, dtype=int)  # [normal_flag, assault, robbery, swoon]

    if pd.isna(cell):
        return vec

    labels = [label.strip() for label in str(cell).split(",")]

    if "normal" in labels:
        vec[0] = 1
    if "assault" in labels:
        vec[1] = 1
    if "robbery" in labels:
        vec[2] = 1
    if "swoon" in labels:
        vec[3] = 1

    return vec

for col in ["y_true", "y_pred_gpt4o", "y_pred_internvl", "y_pred_qwen","y_pred_multi","y_pred_multi2","y_pred_multi3"]:
    df[col] = df[col].apply(map_labels)


y_true = np.stack(df["y_true"].values)
y_true_bin = np.where(y_true[:, 0] == 1, 0, 1)
y_pred_multi = np.stack(df["y_pred_multi"].values)
y_pred_gpt4o = np.stack(df["y_pred_gpt4o"].values)
y_pred_internvl = np.stack(df["y_pred_internvl"].values)
y_pred_qwen = np.stack(df["y_pred_qwen"].values)

y_pred_multi_bin = np.where(y_pred_multi[:, 1:].sum(axis=1) > 0, 1, 0)
y_pred_gpt4o_bin = np.where(y_pred_gpt4o[:, 1:].sum(axis=1) > 0, 1, 0)
y_pred_internvl_bin = np.where(y_pred_internvl[:, 1:].sum(axis=1) > 0, 1, 0)
y_pred_qwen_bin = np.where(y_pred_qwen[:, 1:].sum(axis=1) > 0, 1, 0)

print("=== Multi LLM - Multi label Classification Report (y_pred_multi vs y_true) ===")
print(classification_report(
    y_true,
    y_pred_multi,
    target_names=list(label_to_id.keys()),
    digits=3,
    zero_division=0
))

print("=== Binary Classification Report (normal vs risk, custom rule) ===")
print(classification_report(y_true_bin, y_pred_multi_bin,
                            target_names=["normal", "risk"], digits=3))
print("Accuracy:", accuracy_score(y_true_bin, y_pred_multi_bin))

print("=== Multi LLM - Multi label Classification Report (y_pred_gpt4o vs y_true) ===")
print(classification_report(
    y_true,
    y_pred_gpt4o,
    target_names=list(label_to_id.keys()),
    digits=3,
    zero_division=0
))

print("=== Binary Classification Report (normal vs risk(gpt4o), custom rule) ===")
print(classification_report(y_true_bin, y_pred_gpt4o_bin,
                            target_names=["normal", "risk"], digits=3))
print("Accuracy:", accuracy_score(y_true_bin, y_pred_gpt4o_bin))

print("=== Multi LLM - Multi label Classification Report (y_pred_internvl vs y_true) ===")
print(classification_report(
    y_true,
    y_pred_internvl,
    target_names=list(label_to_id.keys()),
    digits=3,
    zero_division=0
))

print("=== Binary Classification Report (normal vs risk(internvl), custom rule) ===")
print(classification_report(y_true_bin, y_pred_internvl_bin,
                            target_names=["normal", "risk"], digits=3))
print("Accuracy:", accuracy_score(y_true_bin, y_pred_gpt4o_bin))

print("=== Multi LLM - Multi label Classification Report (y_pred_qwen vs y_true) ===")
print(classification_report(
    y_true,
    y_pred_qwen,
    target_names=list(label_to_id.keys()),
    digits=3,
    zero_division=0
))

print("=== Binary Classification Report (normal vs risk(qwen), custom rule) ===")
print(classification_report(y_true_bin, y_pred_qwen_bin,
                            target_names=["normal", "risk"], digits=3))
print("Accuracy:", accuracy_score(y_true_bin, y_pred_gpt4o_bin))


for col in ["y_true", "y_pred_gpt4o", "y_pred_internvl", "y_pred_qwen"]:
    df[col] = df1[col].apply(map_labels_custom)

def to_binary(x):
    return 0 if x == 0 else 1

X = np.stack(df[["y_pred_gpt4o", "y_pred_internvl", "y_pred_qwen"]].apply(lambda row: np.concatenate(row), axis=1).values)
y = np.stack(df["y_true"].values)

X = X.astype(np.int64)
y = y.astype(np.int64)

y = np.array(df["y_true"].tolist())  # (N, 4) one-hot
y = y.astype(np.int64)

print("X shape:", X.shape)  # (N, 12)
print("y shape:", y_true.shape)  # (N, 4)

class DCN_V2(nn.Module):
    def __init__(self, input_dim, num_classes, num_cross=3, low_rank=32, hidden_dim=128):
        super(DCN_V2, self).__init__()
        self.input_dim = input_dim

        # Low-rank Cross Layers
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, low_rank, bias=False),
                nn.Linear(low_rank, input_dim, bias=True)
            ) for _ in range(num_cross)
        ])

        # Deep Layer
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.fc_out = nn.Linear(input_dim + hidden_dim, num_classes)

    def forward(self, x):
        x0 = x
        cross = x
        for layer in self.cross_layers:
            cross = x0 * layer(cross) + cross

        deep_out = self.deep(x0)
        out = torch.cat([cross, deep_out], dim=-1)
        #return torch.sigmoid(self.fc_out(out))
        return self.fc_out(out)

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        # (batch, d)
        xw = torch.sum(xl * self.weight, dim=1, keepdim=True)  # (batch, 1)
        cross = x0 * xw + self.bias + xl
        return cross


class DCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_cross=3, hidden_dim=128, dropout=0.3):
        super(DCN, self).__init__()
        self.num_cross = num_cross
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross)])

        # Deep part (MLP)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output layer
        self.fc_out = nn.Linear(input_dim + hidden_dim, num_classes)

    def forward(self, x):
        x0 = x
        xl = x
        for i in range(self.num_cross):
            xl = self.cross_layers[i](x0, xl)
        deep_out = self.deep(x)
        concat = torch.cat([xl, deep_out], dim=1)
        out = self.fc_out(concat)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = KFold(n_splits=10, shuffle=True, random_state=None)
all_fold_preds = []
all_fold_labels = []
all_fold_probs = []
fold_idx = 1


for train_index, test_index in skf.split(X, y):
    print(f"\n===== Fold {fold_idx} =====")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)  
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)  
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = DCN(input_dim=input_dim, num_classes=num_classes, num_cross=3, hidden_dim=128).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(300):  # 10 epoch
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss: {train_loss/len(train_loader):.4f}")

    model.eval()

    fold_preds, fold_labels, fold_probs = [], [], []
 
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = torch.sigmoid(outputs) > 0.5
            fold_preds.extend(preds.cpu().numpy())
            fold_labels.extend(y_batch.cpu().numpy())
            fold_probs.extend(probs.cpu().numpy())
            

    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)
    all_fold_probs.extend(fold_probs)
    fold_idx += 1


print("\n===== Cross Validation =====")
print(classification_report(all_fold_labels, all_fold_preds, digits=3, target_names=label_to_id.keys()))


multi_label_acc = accuracy_score(all_fold_labels, all_fold_preds)
print("Multi-label Accuracy:", multi_label_acc)

def to_binary(labels):
    return [0 if l == 0 else 1 for l in labels]

def to_binary2(labels, normal_idx=0):
    labels = np.array(labels)
    is_normal = ((labels[:, normal_idx] == 1) & (labels.sum(axis=1) == 1)) | (labels.sum(axis=1) == 0)
    return np.where(is_normal, 0, 1)

y_true_binary = to_binary2(all_fold_labels)
y_pred_binary = to_binary2(all_fold_preds)
y_prob_binary = to_binary2(all_fold_probs)

print(classification_report(y_true_binary, y_pred_binary, digits=3, target_names=["normal", "risk"]))

auc = roc_auc_score(y_true_binary, y_prob_binary)
print("ROC AUC (binary normal vs risk):", auc)


y_true_arr = np.array(all_fold_labels)  
y_prob_arr = np.array(all_fold_probs)   

print("\n===== 클래스별 ROC AUC =====")
for i, cls in enumerate(label_to_id.keys()):
    try:
        auc = roc_auc_score(y_true_arr[:, i], y_prob_arr[:, i])
        print(f"{cls}: AUC = {auc:.3f}")
    except ValueError:
        print(f"{cls}: AUC 계산 불가 (양성/음성 샘플 부족)")
