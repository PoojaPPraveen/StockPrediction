from flask import Flask, request, jsonify, render_template
import joblib
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = torch.load('stock_model.pth')
scaler = joblib.load('scaler.pkl')

class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Reconstruct the model architecture
model_architecture = StockPredictor(input_size=5)
model_architecture.load_state_dict(model)
model_architecture.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [float(data['moving_avg_10']), float(data['moving_avg_50']), 
                float(data['rsi']), float(data['macd']), float(data['daily_return'])]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model_architecture(features_tensor).item()
    
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
