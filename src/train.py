"""
Módulo de Treinamento de Modelos para Previsão de Churn

Este módulo contém funções para treinar e avaliar múltiplos modelos
de classificação, realizar cross-validation e otimizar hiperparâmetros.

Author: Vithor
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """
    Classe responsável pelo treinamento e otimização de modelos.
    
    Atributos:
        models: Dicionário com modelos treinados
        best_model: Melhor modelo encontrado
        X_train, X_test, y_train, y_test: Dados de treino e teste
        results: Resultados das avaliações
    """
    
    def __init__(self, random_state: int = 42):
        """Inicializar o trainer."""
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Dividir dados em treino e teste.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proporção do conjunto de teste
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"✓ Dados divididos em treino (80%) e teste (20%)")
        print(f"  - Treino: {self.X_train.shape[0]} amostras")
        print(f"  - Teste: {self.X_test.shape[0]} amostras")
        print(f"  - Distribuição treino: {self.y_train.value_counts().to_dict()}")
        print(f"  - Distribuição teste: {self.y_test.value_counts().to_dict()}\n")
    
    def apply_smote(self) -> None:
        """
        Aplicar SMOTE para balancear o dataset de treino.
        
        SMOTE (Synthetic Minority Over-sampling Technique) cria amostras sintéticas
        da classe minoritária para balancear o dataset.
        """
        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"✓ SMOTE aplicado ao conjunto de treino")
        print(f"  - Nova distribuição: {self.y_train.value_counts().to_dict()}\n")
    
    def train_baseline_models(self) -> None:
        """
        Treinar modelos baseline com cross-validation.
        
        Modelos treinados:
        - LogisticRegression
        - RandomForestClassifier
        - GradientBoostingClassifier
        """
        print("🤖 Treinando modelos baseline...\n")
        
        # Definir modelos
        models_dict = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1,
                max_depth=5
            )
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model in models_dict.items():
            # Treinar modelo
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')
            
            # Predições no teste
            y_pred_test = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            f1_test = f1_score(self.y_test, y_pred_test)
            auc_test = roc_auc_score(self.y_test, y_pred_proba)
            
            # Armazenar resultados
            self.results[model_name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'f1_score': f1_test,
                'auc_score': auc_test,
                'model': model
            }
            
            print(f"📊 {model_name}:")
            print(f"   CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"   Test F1-Score: {f1_test:.4f}")
            print(f"   Test AUC-ROC: {auc_test:.4f}\n")
        
        # Encontrar melhor modelo
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"🏆 Melhor modelo (baseado em F1-Score): {self.best_model_name}\n")
    
    def optimize_hyperparameters(self) -> None:
        """
        Otimizar hiperparâmetros do melhor modelo usando GridSearchCV.
        """
        print(f"🔍 Otimizando hiperparâmetros do {self.best_model_name}...\n")
        
        # Definir parâmetros conforme o modelo
        if self.best_model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        
        elif self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        
        else:
            print("❌ Modelo não reconhecido para otimização")
            return
        
        # GridSearchCV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Atualizar melhor modelo
        self.best_model = grid_search.best_estimator_
        self.models[self.best_model_name] = self.best_model
        
        print(f"\n✓ Melhores parâmetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"   - {param}: {value}")
        
        print(f"\n✓ Melhor F1-Score no GridSearch: {grid_search.best_score_:.4f}\n")
    
    def train_final_model(self) -> None:
        """
        Treinar o modelo final otimizado no conjunto completo de treino.
        """
        print("🎯 Treinando modelo final com parâmetros otimizados...\n")
        
        self.best_model.fit(self.X_train, self.y_train)
        
        # Avaliação final
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        f1_final = f1_score(self.y_test, y_pred)
        auc_final = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✅ Modelo final treinado!")
        print(f"   F1-Score final: {f1_final:.4f}")
        print(f"   AUC-ROC final: {auc_final:.4f}\n")
    
    def save_model(self, path: str = 'models', name: str = 'churn_model.pkl') -> None:
        """
        Salvar o modelo final e feature names.
        
        Args:
            path (str): Caminho da pasta
            name (str): Nome do arquivo
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, name)
        joblib.dump(self.best_model, model_path)
        
        # Salvar feature names
        feature_names = list(self.X_train.columns)
        features_path = os.path.join(path, 'feature_names.pkl')
        joblib.dump(feature_names, features_path)
        
        print(f"✓ Modelo salvo em '{model_path}'")
        print(f"✓ Feature names salvos em '{features_path}'")
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Retornar resumo dos resultados de treino.
        
        Returns:
            pd.DataFrame: Tabela comparativa dos modelos
        """
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Modelo': model_name,
                'CV F1-Score': f"{results['cv_mean']:.4f}",
                'CV Std': f"{results['cv_std']:.4f}",
                'Test F1-Score': f"{results['f1_score']:.4f}",
                'Test AUC-ROC': f"{results['auc_score']:.4f}"
            })
        
        return pd.DataFrame(summary_data)


def train_churn_model(X: pd.DataFrame, y: pd.Series, apply_smote: bool = True, optimize: bool = True) -> ChurnModelTrainer:
    """
    Função de conveniência para treinar o modelo de churn.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        apply_smote (bool): Se deve aplicar SMOTE
        optimize (bool): Se deve fazer GridSearch
    
    Returns:
        ChurnModelTrainer: Trainer com modelo treinado
    """
    
    trainer = ChurnModelTrainer()
    
    # 1. Dividir dados
    trainer.split_data(X, y)
    
    # 2. Aplicar SMOTE se desejado
    if apply_smote:
        trainer.apply_smote()
    
    # 3. Treinar modelos baseline
    trainer.train_baseline_models()
    
    # 4. Otimizar hiperparâmetros
    if optimize:
        trainer.optimize_hyperparameters()
    
    # 5. Treinar modelo final
    trainer.train_final_model()
    
    # 6. Salvar modelo
    trainer.save_model()
    
    return trainer


if __name__ == "__main__":
    # Exemplo de uso
    from preprocess import ChurnPreprocessor
    
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Pré-processar
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.preprocess(df, training=True)
    preprocessor.save_transformers()
    
    # Treinar
    trainer = train_churn_model(X, y, apply_smote=True, optimize=True)
    
    # Exibir resultados
    print("\n" + "="*60)
    print("RESUMO DOS MODELOS TREINADOS")
    print("="*60)
    print(trainer.get_results_summary())
