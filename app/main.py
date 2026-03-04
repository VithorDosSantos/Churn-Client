"""
API FastAPI para Previsão de Churn de Clientes

Endpoints para fazer predições de churn usando modelo treinado.

Author: Vithor
Date: 2024
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from typing import List

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.predict import ChurnPredictor
from app.schemas import (
    CustomerPredictRequest, CustomerPredictResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, RootResponse, ErrorResponse
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API para prever a probabilidade de churn de clientes em telecomunicações",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variável global para armazenar o preditor
predictor: ChurnPredictor = None
model_loaded: bool = False


def load_model_if_needed():
    """Carrega o modelo se ainda não foi carregado (lazy loading)"""
    global predictor, model_loaded
    
    if model_loaded and predictor is not None:
        return True
    
    try:
        model_path = 'models/churn_model.pkl'
        scaler_path = 'models/scaler.pkl'
        encoders_path = 'models/label_encoders.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders não encontrados: {encoders_path}")
        
        predictor = ChurnPredictor(model_path=model_path)
        model_loaded = True
        logger.info("✅ Modelo carregado com sucesso!")
        return True
    
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        model_loaded = False
        return False


@app.on_event("startup")
async def startup_event():
    """
    Evento de inicialização da API.
    Carrega o modelo e transformers.
    """
    try:
        logger.info("Carregando modelo na inicialização...")
        load_model_if_needed()
    except Exception as e:
        logger.error(f"Erro durante inicialização: {e}")
        model_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento de encerramento da API.
    Limpa recursos se necessário.
    """
    logger.info("API desligando...")


@app.get("/", response_model=RootResponse, tags=["Info"])
async def root():
    """
    Endpoint raiz - Retorna informações sobre a API.
    """
    return RootResponse(
        status="ok",
        model="Churn Predictor v1.0",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Health check - Verifica se o modelo está carregado corretamente.
    """
    if not model_loaded or predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não foi carregado corretamente"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="API está funcionando corretamente"
    )


@app.post("/predict", response_model=CustomerPredictResponse, tags=["Predictions"])
async def predict(request: CustomerPredictRequest):
    """
    Endpoint de predição para um único cliente.
    
    Recebe dados do cliente em JSON e retorna a probabilidade de churn.
    """
    
    # Tentar carregar modelo se necessário
    if not load_model_if_needed():
        logger.error("Tentativa de predição com modelo não carregado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não foi carregado. Tente novamente mais tarde."
        )
    
    try:
        # Converter request para dicionário
        customer_data = request.dict()
        
        logger.info(f"Fazendo predição para cliente: {customer_data}")
        
        # Fazer predição
        result = predictor.predict_single(customer_data)
        
        logger.info(f"Predição concluída: {result}")
        
        return CustomerPredictResponse(**result)
    
    except Exception as e:
        logger.error(f"Erro ao fazer predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Endpoint de predição em lote.
    
    Recebe lista de clientes em JSON e retorna predições para todos.
    """
    
    # Validar entrada primeiro (antes de carregar modelo)
    if not request.customers or len(request.customers) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lista de clientes vazia"
        )
    
    if len(request.customers) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Máximo de 1000 clientes por requisição. Recebido: {len(request.customers)}"
        )
    
    # Tentar carregar modelo se necessário
    if not load_model_if_needed():
        logger.error("Tentativa de predição batch com modelo não carregado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não foi carregado. Tente novamente mais tarde."
        )
    
    try:
        # Converter requests para dicionários
        customers_data = [customer.dict() for customer in request.customers]
        
        logger.info(f"Fazendo predição em lote para {len(customers_data)} clientes")
        
        # Fazer predições
        results = predictor.predict_batch(customers_data)
        
        # Converter para response objects
        predictions = [CustomerPredictResponse(**result) for result in results]
        
        logger.info(f"Predições em lote concluídas: {len(predictions)} clientes")
        
        return BatchPredictionResponse(
            total_customers=len(predictions),
            predictions=predictions
        )
    
    except Exception as e:
        logger.error(f"Erro ao fazer predição em lote: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predições: {str(e)}"
        )


# Manipulador de erros global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manipulador global de exceções."""
    logger.error(f"Erro não tratado: {exc}")
    return {
        "error": "Erro interno do servidor",
        "detail": str(exc),
        "status_code": 500
    }


if __name__ == "__main__":
    # Para desenvolvimento local
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
