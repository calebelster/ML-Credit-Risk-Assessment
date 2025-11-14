import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

CATS = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
df = pd.read_csv(DATA_PATH)
df = df.dropna()
df = pd.get_dummies(df, columns=CATS)
X = df.drop('loan_status', axis=1).values
y = df['loan_status'].values.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

model = CreditRiskNN(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    proba = model(X_test_tensor).numpy().flatten()

results = pd.DataFrame({'y_true': y_test, 'y_pred_prob': proba})
results.to_csv(OUTPUT_PATH / "neural_net_preds.csv", index=False)

print("Neural Net test probabilities:", proba[:5])