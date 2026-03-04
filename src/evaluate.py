"""
Módulo de Avaliação de Modelos para Previsão de Churn

Este módulo contém funções para avaliar modelos, gerar métricas,
gráficos de confusão, ROC curves e feature importance.

Author: Vithor
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import os
from typing import Tuple, Dict, Any


class ChurnModelEvaluator:
    """
    Classe responsável pela avaliação de modelos de churn.
    
    Atributos:
        model: Modelo a ser avaliado
        X_test, y_test: Dados de teste
        y_pred: Predições
        y_pred_proba: Probabilidades de predição
    """
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """Inicializar o avaliador."""
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Criar pasta para gráficos
        os.makedirs('reports/figures', exist_ok=True)
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Executar todas as avaliações do modelo.
        
        Returns:
            Dict: Dicionário com todos os resultados
        """
        results = {}
        
        print("📊 Iniciando avaliação do modelo...\n")
        
        # 1. Classification Report
        results['classification_report'] = self._classification_report()
        
        # 2. Confusion Matrix
        self._confusion_matrix()
        
        # 3. ROC-AUC Curve
        results['auc_score'] = self._roc_auc_curve()
        
        # 4. Feature Importance
        self._feature_importance()
        
        # 5. Threshold Analysis
        self._threshold_analysis()
        
        print("\n✅ Avaliação concluída!")
        
        return results
    
    def _classification_report(self) -> str:
        """
        Gerar relatório de classificação completo.
        
        Returns:
            str: Relatório de classificação
        """
        print("="*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*60)
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['Sem Churn', 'Com Churn'],
            digits=4
        )
        print(report)
        
        # Métricas adicionais
        f1 = f1_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        
        print("\n" + "="*60)
        print("MÉTRICAS RESUMIDAS")
        print("="*60)
        print(f"F1-Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print("="*60 + "\n")
        
        return report
    
    def _confusion_matrix(self) -> None:
        """
        Plotar matriz de confusão.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Sem Churn', 'Com Churn'],
                    yticklabels=['Sem Churn', 'Com Churn'],
                    cbar_kws={'label': 'Frequência'})
        plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/figures/08_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Matriz de confusão salva em 'reports/figures/08_confusion_matrix.png'")
    
    def _roc_auc_curve(self) -> float:
        """
        Plotar curva ROC-AUC.
        
        Returns:
            float: Score AUC
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC-AUC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/09_roc_auc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Curva ROC-AUC salva em 'reports/figures/09_roc_auc_curve.png'")
        print(f"  AUC-ROC: {auc_score:.4f}")
        
        return auc_score
    
    def _feature_importance(self) -> None:
        """
        Plotar importância das features (para modelos que suportam).
        """
        # Verificar se o modelo tem atributo feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X_test.columns
            
            # Ordenar
            indices = np.argsort(importances)[::-1][:15]  # Top 15
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), feature_names[indices])
            plt.xlabel('Importância', fontsize=12)
            plt.title('Top 15 Features Mais Importantes', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('reports/figures/10_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Feature Importance salva em 'reports/figures/10_feature_importance.png'")
            
            # Exibir top features
            print("\nTop 15 Features Mais Importantes:")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
        
        elif hasattr(self.model, 'coef_'):
            # Para LogisticRegression
            coefs = np.abs(self.model.coef_[0])
            feature_names = self.X_test.columns
            
            # Ordenar
            indices = np.argsort(coefs)[::-1][:15]  # Top 15
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), coefs[indices], color='steelblue')
            plt.yticks(range(len(indices)), feature_names[indices])
            plt.xlabel('Coeficiente Absoluto', fontsize=12)
            plt.title('Top 15 Features com Maior Coeficiente', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('reports/figures/10_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Feature Importance (Coeficientes) salva em 'reports/figures/10_feature_importance.png'")
            
            # Exibir top features
            print("\nTop 15 Features com Maior Coeficiente:")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:2d}. {feature_names[idx]:30s} {coefs[idx]:.4f}")
        
        else:
            print("⚠️  Modelo não suporta análise de feature importance")
    
    def _threshold_analysis(self) -> None:
        """
        Análise de como precision e recall variam com threshold.
        """
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        print("\n" + "="*60)
        print("ANÁLISE DE THRESHOLD")
        print("="*60)
        print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*60)
        
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred_threshold = (self.y_pred_proba >= threshold).astype(int)
            precision = precision_score(self.y_test, y_pred_threshold, zero_division=0)
            recall = recall_score(self.y_test, y_pred_threshold, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_threshold, zero_division=0)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            print(f"{threshold:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        print("="*60 + "\n")
        
        # Plotar
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision_scores, marker='o', label='Precision', linewidth=2)
        plt.plot(thresholds, recall_scores, marker='s', label='Recall', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision e Recall vs Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/11_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Análise de threshold salva em 'reports/figures/11_threshold_analysis.png'")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Função de conveniência para avaliar um modelo.
    
    Args:
        model: Modelo treinado
        X_test (pd.DataFrame): Features de teste
        y_test (pd.Series): Target de teste
    
    Returns:
        Dict: Resultados da avaliação
    """
    evaluator = ChurnModelEvaluator(model, X_test, y_test)
    results = evaluator.evaluate_all()
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    import joblib
    from preprocess import ChurnPreprocessor
    
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Pré-processar
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.preprocess(df, training=False)
    preprocessor.load_transformers()
    
    # Carregar modelo
    model = joblib.load('../models/churn_model.pkl')
    
    # Dividir dados
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Avaliar
    evaluate_model(model, X_test, y_test)
