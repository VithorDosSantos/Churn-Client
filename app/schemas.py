"""
Esquemas Pydantic para Validação de Dados da API

Define os modelos de requisição e resposta para os endpoints da API.

Author: Vithor
Date: 2024
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerPredictRequest(BaseModel):
    """Schema para requisição de predição de um cliente."""
    
    tenure: int = Field(..., ge=0, le=72, description="Meses de contrato (0-72)")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Cobrana mensal em $")
    TotalCharges: float = Field(..., ge=0, description="Cobrana total em $")
    Contract: str = Field(..., description="Tipo de contrato: Month-to-month, One year, Two year")
    PaymentMethod: str = Field(..., description="Método de pagamento")
    InternetService: str = Field(..., description="Tipo de serviço: Fiber optic, DSL, No")
    gender: str = Field(..., description="Gênero: Male, Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="É cidadão sênior? (0 ou 1)")
    Partner: str = Field(..., description="Tem parceiro? Yes ou No")
    Dependents: str = Field(..., description="Tem dependentes? Yes ou No")
    PhoneService: str = Field(..., description="Tem serviço de telefone? Yes ou No")
    PaperlessBilling: str = Field(..., description="Fatura sem papel? Yes ou No")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class CustomerPredictResponse(BaseModel):
    """Schema para resposta de predição."""
    
    churn_prediction: int = Field(..., description="Predição: 0 (Sem churn) ou 1 (Com churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probabilidade de churn (0-1)")
    risk_level: str = Field(..., description="Nível de risco: Baixo, Médio ou Alto")
    message: str = Field(..., description="Mensagem descritiva")
    
    class Config:
        json_schema_extra = {
            "example": {
                "churn_prediction": 1,
                "churn_probability": 0.83,
                "risk_level": "Alto",
                "message": "Cliente com alta probabilidade de cancelamento"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema para requisição de predição em lote."""
    
    customers: List[CustomerPredictRequest] = Field(..., description="Lista de clientes para prediçao")
    
    class Config:
        json_schema_extra = {
            "example": {
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
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema para resposta de predição em lote."""
    
    total_customers: int = Field(..., description="Total de clientes processados")
    predictions: List[CustomerPredictResponse] = Field(..., description="Lista de predições")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_customers": 1,
                "predictions": [
                    {
                        "churn_prediction": 1,
                        "churn_probability": 0.83,
                        "risk_level": "Alto",
                        "message": "Cliente com alta probabilidade de cancelamento"
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    
    status: str = Field(..., description="Status: healthy ou unhealthy")
    model_loaded: bool = Field(..., description="Modelo foi carregado com sucesso?")
    message: str = Field(..., description="Mensagem descritiva")


class RootResponse(BaseModel):
    """Schema para resposta da raiz."""
    
    status: str = Field(..., description="Status: ok")
    model: str = Field(..., description="Nome e versão do modelo")
    version: str = Field(..., description="Versão da API")


class ErrorResponse(BaseModel):
    """Schema para resposta de erro."""
    
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes do erro")
    status_code: int = Field(..., description="Código HTTP")
