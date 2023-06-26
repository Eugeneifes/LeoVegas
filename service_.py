from flask import Flask, request, jsonify, abort
from flasgger import Swagger, swag_from
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


app = Flask(__name__)
swagger = Swagger(app)

api_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 4
hidden_size = 64
output_size = 1

model_path = 'deposit_next_NN.pt'
model = NeuralNet(input_size, hidden_size, output_size)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model_info = {
    'model': model,
    'version': 1,
    'test_loss': 4190266.5
}


@app.route('/predict', methods=['POST'])
@swag_from('docs/predict.yml')
def predict():
    try:
        data = request.get_json()

        if 'values' not in data or 'api-key' not in data:
            raise ValueError('Invalid request data. Missing parameters.')

        values = data['values']
        api_key = data['api-key']

        # Validate API Key
        if api_key != api_token:
            raise ValueError('Invalid API key.')

        # Validate 'values' parameter
        if not isinstance(values, list):
            raise ValueError('Invalid data format. Expected a list of lists.')

        for row in values:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError('Invalid number of values. Each row must contain exactly 4 values.')

            for value in row:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError('Invalid value. Each value must be numeric and non-negative.')

        input_data = pd.DataFrame(values, columns=['tenure', 'deposit', 'turnover', 'withdrawal'])
        model.eval()
        predictions = model(torch.tensor(input_data.values, dtype=torch.float32)).squeeze().tolist()

        if not isinstance(predictions, list):
            predictions = [predictions]

        return jsonify({'predictions': predictions})

    except ValueError as e:
        operation_status = {'status': str(e)}
        return jsonify(operation_status), 400

    except Exception as e:
        operation_status = {'status': 'Error occurred while processing the request.'}
        return jsonify(operation_status), 400


@app.route('/health_check', methods=['GET'])
@swag_from('docs/health_check.yml')
def health_check():
    health_status = {'status': 'healthy', 'model version': model_info['version'], 'test_loss': model_info['test_loss']}
    return jsonify(health_status)


@app.route('/add_data', methods=['POST'])
@swag_from('docs/add_data.yml')
def add_data():
    try:
        data = request.get_json()

        if 'values' not in data or 'api-key' not in data:
            raise ValueError('Invalid request data. Missing parameters.')

        values = data['values']
        api_key = data['api-key']

        # Validate API Key
        if api_key != api_token:
            raise ValueError('Invalid API key.')

        # Validate 'values' parameter
        if not isinstance(values, list):
            raise ValueError('Invalid data format. Expected a list of lists.')

        for row in values:
            if not isinstance(row, list) or len(row) != 5:
                raise ValueError('Invalid number of values. Each row must contain exactly 5 values.')

            for value in row:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError('Invalid value. Each value must be numeric and non-negative.')

        try:
            df = pd.read_csv('new_data.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['tenure', 'deposit', 'turnover', 'withdrawal', 'deposit_next'])

        new_data = pd.DataFrame(values, columns=['tenure', 'deposit', 'turnover', 'withdrawal', 'deposit_next'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv('new_data.csv', index=False)

        operation_status = {'status': 'Data added successfully.'}
        return jsonify(operation_status)

    except ValueError as e:
        operation_status = {'status': str(e)}
        return jsonify(operation_status), 400

    except Exception as e:
        operation_status = {'status': 'Error occurred while processing the request.'}
        return jsonify(operation_status), 400


@app.route('/retrain', methods=['POST'])
@swag_from('docs/retrain.yml')
def retrain():
    try:
        data = request.get_json()

        if 'api-key' not in data:
            raise ValueError('Invalid request data. Missing parameters.')

        api_key = data['api-key']

        # Validate API Key
        if api_key != api_token:
            raise ValueError('Invalid API key.')

        # Check if new_data.csv file exists and is not empty
        if not os.path.isfile('new_data.csv') or os.stat('new_data.csv').st_size == 0:
            operation_status = {'status': 'No data for training.'}
            return jsonify(operation_status)

        df = pd.read_csv('new_data.csv')
        x = df[['tenure', 'deposit', 'turnover', 'withdrawal']].values
        y = df['deposit_next'].values
        x_train_tensor = torch.Tensor(x)
        y_train_tensor = torch.Tensor(y)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        learning_rate = 0.001
        num_epochs = 100

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Delete new_data.csv file
        os.remove('new_data.csv')
        model_info['model'] = model
        model_info['version'] += 1

        operation_status = {'status': f"Model retrained, new version: {model_info['version']}."}
        return jsonify(operation_status)

    except ValueError as e:
        operation_status = {'status': str(e)}
        return jsonify(operation_status), 400

    except Exception as e:
        operation_status = {'status': 'Error occurred while processing the request.'}
        return jsonify(operation_status), 400


@app.route('/evaluate_model', methods=['POST'])
@swag_from('docs/evaluate_model.yml')
def evaluate_model():
    try:
        data = request.get_json()

        if 'api-key' not in data:
            raise ValueError('Invalid request data. Missing parameters.')

        api_key = data['api-key']

        # Validate API Key
        if api_key != api_token:
            raise ValueError('Invalid API key.')

        # Check if new_data.csv file exists and is not empty
        if not os.path.isfile('new_data.csv') or os.stat('new_data.csv').st_size == 0:
            operation_status = {'status': 'No data for validation.'}
            return jsonify(operation_status)

        df = pd.read_csv('new_data.csv')
        x = df[['tenure', 'deposit', 'turnover', 'withdrawal']].values
        y = df['deposit_next'].values
        x_tensor = torch.Tensor(x)
        y_tensor = torch.Tensor(y)

        predictions = model(x_tensor).squeeze()
        loss = nn.MSELoss()(predictions, y_tensor)

        operation_status = {'loss': loss.item(), 'test_loss': model_info['test_loss'], 'model_version': model_info['version']}
        return jsonify(operation_status)

    except ValueError as e:
        operation_status = {'status': str(e)}
        return jsonify(operation_status), 400

    except Exception as e:
        operation_status = {'status': 'Error occurred while processing the request.'}
        return jsonify(operation_status), 400


if __name__ == '__main__':
    app.run(debug=True)
