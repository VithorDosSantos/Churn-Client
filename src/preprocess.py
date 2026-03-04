"""
Módulo de Pré-processamento de Dados para Previsão de Churn

Este módulo contém funções para limpar, transformar e normalizar dados
do dataset Telco Customer Churn, preparando-os para treinamento de modelos.

Author: Vithor
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from typing import Tuple, Dict, Any


class ChurnPreprocessor:
    """
    Classe responsável pelo pré-processamento dos dados de churn.
    
    Atributos:
        scaler: StandardScaler para normalizar variáveis numéricas
        label_encoders: Dicionário com LabelEncoders para cada variável categórica
        one_hot_cols: Colunas que recebem One-Hot Encoding
    """
    
    def __init__(self):
        """Inicializar o preprocessador."""
        self.scaler = None
        self.label_encoders = {}
        self.one_hot_cols = []
        self.numeric_cols = []
        self.binary_cols = []
        
    def preprocess(self, df: pd.DataFrame, training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica todo o pipeline de pré-processamento nos dados.
        
        Etapas:
        1. Remove coluna customerID (não é feature)
        2. Converte TotalCharges para numérico
        3. Preenche NaN de TotalCharges com a mediana
        4. Converte Churn de "Yes"/"No" para 1/0
        5. Aplica Label Encoding em variáveis binárias
        6. Aplica One-Hot Encoding em variáveis com mais de 2 categorias
        7. Aplica StandardScaler em variáveis numéricas
        
        Args:
            df (pd.DataFrame): DataFrame com dados brutos
            training (bool): Se True, treina encoders/scaler. Se False, apenas aplica.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) e Target (y)
        
        Raises:
            ValueError: Se colunas obrigatórias estão faltando
        """
        
        df = df.copy()
        
        print("🔧 Iniciando pré-processamento...")
        
        # 1. Remover customerID
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)
            print("  ✓ Removida coluna customerID")
        
        # 2 & 3. Converter e preencher TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            median_total_charges = df['TotalCharges'].median()
            df['TotalCharges'].fillna(median_total_charges, inplace=True)
            print(f"  ✓ TotalCharges convertido e NaNs preenchidos (mediana: {median_total_charges})")
        
        # 4. Converter Churn para 1/0
        if 'Churn' in df.columns:
            y = (df['Churn'] == 'Yes').astype(int)
            df.drop('Churn', axis=1, inplace=True)
            print("  ✓ Churn convertido para 1/0")
        else:
            raise ValueError("Coluna 'Churn' não encontrada no dataset")
        
        # Identificar tipos de variáveis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 5. Label Encoding para variáveis binárias
        binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
        
        if training:
            print(f"  • Variáveis binárias encontradas: {binary_cols}")
        
        for col in binary_cols:
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        if binary_cols:
            print(f"  ✓ Label Encoding aplicado em {len(binary_cols)} variáveis binárias")
        
        # 6. One-Hot Encoding para variáveis com mais de 2 categorias
        multi_cat_cols = [col for col in categorical_cols if df[col].nunique() > 2 and col not in binary_cols]
        
        if training:
            self.one_hot_cols = multi_cat_cols
            print(f"  • Variáveis para One-Hot Encoding: {multi_cat_cols}")
        
        if multi_cat_cols:
            df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
            print(f"  ✓ One-Hot Encoding aplicado em {len(multi_cat_cols)} variáveis")
        
        # Atualizar lista de colunas numéricas após transformações
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 7. StandardScaler em variáveis numéricas
        if training:
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.numeric_cols = numeric_cols
            print(f"  ✓ StandardScaler aplicado em {len(numeric_cols)} variáveis numéricas")
        else:
            if self.scaler is not None:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            else:
                raise ValueError("Scaler não foi inicializado. Treinar antes de aplicar dados novos.")
        
        print("✅ Pré-processamento concluído!\n")
        
        return df, y
    
    def save_transformers(self, path: str = 'models') -> None:
        """
        Salva o scaler e label encoders para reutilização.
        
        Args:
            path (str): Caminho da pasta para salvar os transformers
        """
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.pkl'))
        
        # Salvar metadados
        metadata = {
            'numeric_cols': self.numeric_cols,
            'label_encoded_cols': list(self.label_encoders.keys()),
            'one_hot_cols': self.one_hot_cols
        }
        joblib.dump(metadata, os.path.join(path, 'preprocess_metadata.pkl'))
        
        print(f"✓ Transformers salvos em '{path}/'")
    
    def load_transformers(self, path: str = 'models') -> None:
        """
        Carrega o scaler e label encoders anteriormente salvos.
        
        Args:
            path (str): Caminho da pasta onde os transformers estão salvos
        """
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.pkl'))
        
        metadata = joblib.load(os.path.join(path, 'preprocess_metadata.pkl'))
        self.numeric_cols = metadata['numeric_cols']
        self.one_hot_cols = metadata['one_hot_cols']
        
        print(f"✓ Transformers carregados de '{path}/'")


def preprocess(df: pd.DataFrame, training: bool = True, preprocessor: ChurnPreprocessor = None) -> Tuple[pd.DataFrame, pd.Series, ChurnPreprocessor]:
    """
    Função de conveniência para pré-processamento.
    
    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        training (bool): Se True, cria novo preprocessador. Se False, usa o existente.
        preprocessor (ChurnPreprocessor): Preprocessador existente (se training=False)
    
    Returns:
        Tuple: (X, y, preprocessor)
    """
    
    if preprocessor is None:
        preprocessor = ChurnPreprocessor()
    
    X, y = preprocessor.preprocess(df, training=training)
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Exemplo de uso
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    preprocessor = ChurnPreprocessor()
    X_train, y_train = preprocessor.preprocess(df, training=True)
    
    print(f"\nResultado:")
    print(f"  Shape de X: {X_train.shape}")
    print(f"  Shape de y: {y_train.shape}")
    print(f"  Distribuição de y: {y_train.value_counts()}")
    
    # Salvar transformers
    preprocessor.save_transformers()
