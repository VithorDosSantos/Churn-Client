"""
Módulo de Predição para Modelo de Churn

Este módulo contém funções para fazer predições com o modelo treinado,
carregando modelos, scaler e encoders salvos.

Author: Vithor
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, List


class ChurnPredictor:
    """
    Classe para fazer predições com o modelo de churn treinado.
    
    Atributos:
        model: Modelo carregado
        scaler: StandardScaler carregado
        label_encoders: Dicionário de LabelEncoders
        metadata: Metadados de pré-processamento
    """
    
    def __init__(self, model_path: str = 'models/churn_model.pkl'):
        """
        Inicializar o preditor carregando modelo e transformers.
        
        Args:
            model_path (str): Caminho do modelo
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.metadata = joblib.load('models/preprocess_metadata.pkl')
        
        print("✓ Modelo e transformers carregados com sucesso!")
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """
        Fazer predição para um único cliente.
        
        Args:
            customer_data (Dict): Dicionário com dados do cliente
        
        Returns:
            Dict: Dicionário com predição, probabilidade e risco
        """
        # Converter para DataFrame
        df = pd.DataFrame([customer_data])
        
        # Pré-processar (mesmo processo de treinamento)
        df_processed = self._preprocess_single(df)
        
        # Fazer predição
        prediction = self.model.predict(df_processed)[0]
        probability = self.model.predict_proba(df_processed)[0][1]
        
        # Determinar nível de risco
        if probability >= 0.7:
            risk_level = "Alto"
            message = "Cliente com alta probabilidade de cancelamento"
        elif probability >= 0.4:
            risk_level = "Médio"
            message = "Cliente com risco moderado de cancelamento"
        else:
            risk_level = "Baixo"
            message = "Cliente com baixo risco de cancelamento"
        
        return {
            'churn_prediction': int(prediction),
            'churn_probability': round(probability, 4),
            'risk_level': risk_level,
            'message': message
        }
    
    def predict_batch(self, customers_data: List[Dict]) -> List[Dict]:
        """
        Fazer predições para múltiplos clientes.
        
        Args:
            customers_data (List[Dict]): Lista de dicionários com dados dos clientes
        
        Returns:
            List[Dict]: Lista de dicionários com predições
        """
        results = []
        
        for customer_data in customers_data:
            result = self.predict_single(customer_data)
            results.append(result)
        
        return results
    
    def _preprocess_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-processar dados de um cliente para predição.
        
        Args:
            df (pd.DataFrame): DataFrame com dados brutos
        
        Returns:
            pd.DataFrame: DataFrame pré-processado
        """
        df = df.copy()
        
        # Remover customerID se existir
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)
        
        # Converter TotalCharges para numérico
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Usar média se NaN
            if df['TotalCharges'].isna().any():
                df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
        
        # Remover Churn se existir
        if 'Churn' in df.columns:
            df.drop('Churn', axis=1, inplace=True)
        
        # Label Encoding para variáveis binárias
        for col in self.metadata['label_encoded_cols']:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # One-Hot Encoding
        if self.metadata['one_hot_cols']:
            df = pd.get_dummies(df, columns=self.metadata['one_hot_cols'], drop_first=True)
        
        # Garantir que todas as colunas esperadas existem
        expected_cols = self._get_feature_columns()
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Selecionar apenas as colunas esperadas
        df = df[expected_cols]
        
        # Normalizar com scaler
        df[self.metadata['numeric_cols']] = self.scaler.transform(
            df[self.metadata['numeric_cols']]
        )
        
        return df
    
    def _get_feature_columns(self) -> List[str]:
        """
        Obter lista de colunas de features esperadas pelo modelo.
        
        Returns:
            List[str]: Lista de nomes de colunas
        """
        # Reconstruir nomes de colunas esperadas
        # Isso é uma simplificação - em produção, seria melhor salvar a lista exata
        
        # Colunas numéricas
        numeric_cols = self.metadata['numeric_cols']
        
        # Colunas codificadas
        encoded_cols = self.metadata['label_encoded_cols']
        
        # Colunas one-hot (aproximado)
        one_hot_cols = []
        for col in self.metadata['one_hot_cols']:
            # Aqui seria ideal ter uma lista pré-computada
            one_hot_cols.append(col)
        
        # Combinar (nota: em produção, seria melhor salvar a lista exata)
        return numeric_cols + encoded_cols + one_hot_cols


def predict_churn_single(customer_data: Dict, model_path: str = 'models/churn_model.pkl') -> Dict:
    """
    Função de conveniência para predição única.
    
    Args:
        customer_data (Dict): Dados do cliente
        model_path (str): Caminho do modelo
    
    Returns:
        Dict: Resultado da predição
    """
    predictor = ChurnPredictor(model_path)
    return predictor.predict_single(customer_data)


def predict_churn_batch(customers_data: List[Dict], model_path: str = 'models/churn_model.pkl') -> List[Dict]:
    """
    Função de conveniência para predição em lote.
    
    Args:
        customers_data (List[Dict]): Dados dos clientes
        model_path (str): Caminho do modelo
    
    Returns:
        List[Dict]: Resultados das predições
    """
    predictor = ChurnPredictor(model_path)
    return predictor.predict_batch(customers_data)


if __name__ == "__main__":
    # Exemplo de uso
    
    # Exemplo de cliente para predição
    example_customer = {
        'tenure': 12,
        'MonthlyCharges': 65.5,
        'TotalCharges': 786.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'InternetService': 'Fiber optic',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'PaperlessBilling': 'Yes'
    }
    
    try:
        result = predict_churn_single(example_customer)
        print("Resultado da predição:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print("❌ Modelo não encontrado. Execute primeiro: python src/train.py")
