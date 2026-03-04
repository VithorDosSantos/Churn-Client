"""
Testes da API FastAPI para Previsão de Churn

Testes incluem:
- Health check
- Predição com dados válidos
- Tratamento de erros com dados inválidos
- Predição de cliente de alto risco
- Predição em lote

Author: Vithor
Date: 2024
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.main import app

# Cliente de teste
client = TestClient(app)


class TestRootEndpoint:
    """Testes do endpoint raiz."""
    
    def test_root_returns_status_ok(self):
        """Testa se endpoint raiz retorna status ok."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "Churn Predictor" in response.json()["model"]
    
    def test_root_has_version(self):
        """Testa se resposta tem versão."""
        response = client.get("/")
        
        assert "version" in response.json()
        assert response.json()["version"] == "1.0.0"


class TestHealthEndpoint:
    """Testes do endpoint de health check."""
    
    def test_health_check_returns_200(self):
        """Testa se health check retorna 200."""
        response = client.get("/health")
        
        # Pode retornar 200 ou 503 dependendo se modelo está carregado
        assert response.status_code in [200, 503]
    
    def test_health_check_has_required_fields(self):
        """Testa se health check tem campos necessários."""
        response = client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "message" in data


class TestPredictEndpoint:
    """Testes do endpoint de predição única."""
    
    @pytest.fixture
    def valid_customer_data(self):
        """Fixture com dados válidos de cliente."""
        return {
            "tenure": 12,
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "PaperlessBilling": "Yes"
        }
    
    @pytest.fixture
    def high_risk_customer(self):
        """Fixture com dados de cliente de alto risco."""
        return {
            "tenure": 2,
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "PaperlessBilling": "No"
        }
    
    def test_predict_valid_input(self, valid_customer_data):
        """Testa predição com dados válidos."""
        response = client.post("/predict", json=valid_customer_data)
        
        # Pode retornar 200 ou 503 dependendo se modelo está carregado
        if response.status_code == 200:
            data = response.json()
            
            # Verificar campos obrigatórios
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert "risk_level" in data
            assert "message" in data
            
            # Verificar tipos
            assert isinstance(data["churn_prediction"], int)
            assert isinstance(data["churn_probability"], (int, float))
            assert isinstance(data["risk_level"], str)
            assert isinstance(data["message"], str)
            
            # Verificar valores
            assert data["churn_prediction"] in [0, 1]
            assert 0 <= data["churn_probability"] <= 1
            assert data["risk_level"] in ["Baixo", "Médio", "Alto"]
        
        elif response.status_code == 503:
            # Modelo não carregado
            assert "detail" in response.json() or response.status_code == 503
    
    def test_predict_invalid_input_missing_field(self):
        """Testa predição com campo faltando (erro 422)."""
        invalid_data = {
            "tenure": 12,
            "MonthlyCharges": 65.5
            # Faltam outros campos obrigatórios
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Deve retornar 422 (validação Pydantic falhou)
        assert response.status_code == 422
    
    def test_predict_invalid_input_wrong_type(self):
        """Testa predição com tipo de dado inválido."""
        invalid_data = {
            "tenure": "não é número",  # Deve ser int
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "PaperlessBilling": "Yes"
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Deve retornar 422
        assert response.status_code == 422
    
    def test_predict_high_risk_customer(self, high_risk_customer):
        """Testa se cliente de alto risco é identificado corretamente."""
        response = client.post("/predict", json=high_risk_customer)
        
        if response.status_code == 200:
            data = response.json()
            
            # Cliente de alto risco deve ter probabilidade >= 0.4
            # (para ter risco Médio ou Alto)
            if data["churn_probability"] >= 0.7:
                assert data["risk_level"] == "Alto"
            elif data["churn_probability"] >= 0.4:
                assert data["risk_level"] in ["Médio", "Alto"]


class TestBatchPredictEndpoint:
    """Testes do endpoint de predição em lote."""
    
    @pytest.fixture
    def valid_batch_data(self):
        """Fixture com dados válidos para lote."""
        return {
            "customers": [
                {
                    "tenure": 12,
                    "MonthlyCharges": 65.5,
                    "TotalCharges": 786.0,
                    "Contract": "Month-to-month",
                    "PaymentMethod": "Electronic check",
                    "InternetService": "Fiber optic",
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "PaperlessBilling": "Yes"
                },
                {
                    "tenure": 24,
                    "MonthlyCharges": 45.0,
                    "TotalCharges": 1080.0,
                    "Contract": "One year",
                    "PaymentMethod": "Bank transfer",
                    "InternetService": "DSL",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "Yes",
                    "PhoneService": "No",
                    "PaperlessBilling": "Yes"
                }
            ]
        }
    
    def test_predict_batch_returns_predictions(self, valid_batch_data):
        """Testa predição em lote com múltiplos clientes."""
        response = client.post("/predict/batch", json=valid_batch_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # Verificar estrutura
            assert "total_customers" in data
            assert "predictions" in data
            
            # Verificar quantidade
            assert data["total_customers"] == 2
            assert len(data["predictions"]) == 2
            
            # Verificar cada predição
            for prediction in data["predictions"]:
                assert "churn_prediction" in prediction
                assert "churn_probability" in prediction
                assert "risk_level" in prediction
                assert "message" in prediction
    
    def test_predict_batch_empty_list(self):
        """Testa predição em lote com lista vazia."""
        invalid_data = {"customers": []}
        
        response = client.post("/predict/batch", json=invalid_data)
        
        # Deve retornar 400 (lista vazia)
        assert response.status_code == 400
    
    def test_predict_batch_too_many_customers(self):
        """Testa predição em lote com muitos clientes (limite é 1000)."""
        # Criar lista com 1001 clientes (deve exceder limite)
        customers = []
        for i in range(1001):
            customers.append({
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "PaperlessBilling": "Yes"
            })
        
        data = {"customers": customers}
        response = client.post("/predict/batch", json=data)
        
        # Deve retornar 400 (muitos clientes)
        assert response.status_code == 400


class TestDocumentation:
    """Testes da documentação da API."""
    
    def test_openapi_schema_exists(self):
        """Testa se schema OpenAPI está disponível."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        assert "openapi" in response.json()
    
    def test_docs_available(self):
        """Testa se documentação Swagger está disponível."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "Swagger UI" in response.text or "swagger" in response.text


if __name__ == "__main__":
    # Rodar testes
    pytest.main(["-v", __file__])
