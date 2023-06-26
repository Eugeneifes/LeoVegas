import pytest
from service_ import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get('/health_check')
    assert response.status_code == 200
    assert response.json == {'status': 'healthy', 'model version': 1, 'test_loss': 4190266.5}


def test_predict(client):
    # Prepare test data
    data = {
        'values': [[12, 770, 1650, 1400], [15, 1000, 2000, 500], [20, 500, 1500, 1000]],
        'api-key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
    }

    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert response.json == {"predictions": [55.950653076171875, 21.68572235107422, 33.31529998779297]}

def test_add_data(client):
    # Prepare test data
    data = {
        'values': [[12,770,1650,1400,2820],[15,1000,2000,500,2300],[20,500,1500,1000,300]],
        'api-key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
    }

    response = client.post('/add_data', json=data)
    assert response.status_code == 200
    assert response.json == {'status': 'Data added successfully.'}


def test_evaluate_model(client):
    data = {
        'api-key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
    }

    response = client.post('/evaluate_model', json=data)
    assert response.status_code == 200
    assert response.json == {'loss': 4300601.5, 'test_loss': 4190266.5, 'model_version': 1}


def test_retrain(client):
    data = {
        'api-key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
    }

    response = client.post('/retrain', json=data)
    assert response.status_code == 200
    assert response.json == {'status': f"Model retrained, new version: 2."}


if __name__ == '__main__':
    pytest.main()
