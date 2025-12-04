import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PredictiveMaintenance:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.clean_data()
        
    def clean_data(self):
        """Limpieza y preparación de datos"""
        # Eliminar última fila incompleta
        self.df = self.df[:-1]
        
        # Convertir columnas a numéricas
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Encoding de variable Type
        self.le = LabelEncoder()
        self.df['Type_encoded'] = self.le.fit_transform(self.df['Type'])
        
    def exploratory_analysis(self):
        """Análisis exploratorio de datos"""
        print("=" * 50)
        print("ANÁLISIS EXPLORATORIO")
        print("=" * 50)
        
        # Información básica
        print(f"📊 Dimensiones del dataset: {self.df.shape}")
        print(f"🔧 Fallas detectadas: {self.df['Machine failure'].sum()} ({self.df['Machine failure'].mean()*100:.1f}%)")
        
        # Distribución de tipos
        print("\n📈 Distribución por Tipo:")
        print(self.df['Type'].value_counts())
        
        # Estadísticas descriptivas
        print("\n📋 Estadísticas de variables numéricas:")
        print(self.df[['Air temperature [K]', 'Process temperature [K]', 
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].describe())
        
    def visualize_data(self):
        """Visualizaciones del dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de fallas
        self.df['Machine failure'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0,0])
        axes[0,0].set_title('Distribución de Fallas')
        
        # 2. Tool wear vs Fallas
        sns.boxplot(data=self.df, x='Machine failure', y='Tool wear [min]', ax=axes[0,1])
        axes[0,1].set_title('Tool Wear vs Fallas')
        
        # 3. Torque vs Fallas
        sns.boxplot(data=self.df, x='Machine failure', y='Torque [Nm]', ax=axes[0,2])
        axes[0,2].set_title('Torque vs Fallas')
        
        # 4. Tipo de producto vs Fallas
        pd.crosstab(self.df['Type'], self.df['Machine failure']).plot.bar(ax=axes[1,0])
        axes[1,0].set_title('Tipo de Producto vs Fallas')
        
        # 5. Correlación de variables
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                       'Machine failure']
        sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Matriz de Correlación')
        
        # 6. Rotational speed vs Torque (coloreado por fallas)
        scatter = axes[1,2].scatter(self.df['Rotational speed [rpm]'], 
                                  self.df['Torque [Nm]'], 
                                  c=self.df['Machine failure'], 
                                  alpha=0.6, cmap='viridis')
        axes[1,2].set_xlabel('Rotational Speed [rpm]')
        axes[1,2].set_ylabel('Torque [Nm]')
        axes[1,2].set_title('Speed vs Torque (Fallas en amarillo)')
        plt.colorbar(scatter, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig('analisis_fallas.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_features(self):
        """Preparación de características para el modelo"""
        # Selección de features
        features = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                   'Type_encoded']
        
        X = self.df[features]
        y = self.df['Machine failure']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Escalado
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Manejo de desbalanceo con SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, features
    
    def train_models(self):
        """Entrenamiento de modelos de ML"""
        X_train, X_test, y_train, y_test, scaler, features = self.prepare_features()
        
        print("=" * 50)
        print("ENTRENAMIENTO DE MODELOS")
        print("=" * 50)
        
        # Modelos a probar
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🏋️‍♂️ Entrenando {name}...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.3f} - AUC: {auc_score:.3f}")
            
            # Reporte de clasificación
            print(f"\n📊 Reporte de Clasificación - {name}:")
            print(classification_report(y_test, y_pred))
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusión - {name}')
            plt.ylabel('Real')
            plt.xlabel('Predicho')
            plt.savefig(f'matriz_confusion_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        self.results = results
        self.feature_names = features
        self.scaler = scaler
        
        return results
    
    def feature_importance(self):
        """Análisis de importancia de características"""
        if hasattr(self, 'results'):
            rf_model = self.results['Random Forest']['model']
            
            # Importancia de features
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("=" * 50)
            print("IMPORTANCIA DE CARACTERÍSTICAS")
            print("=" * 50)
            print(importance)
            
            # Visualización
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance, x='importance', y='feature')
            plt.title('Importancia de Características - Random Forest')
            plt.xlabel('Importancia')
            plt.tight_layout()
            plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance
    
    def risk_assessment_system(self):
        """Sistema de evaluación de riesgo"""
        print("=" * 50)
        print("SISTEMA DE EVALUACIÓN DE RIESGO")
        print("=" * 50)
        
        # Reglas simples basadas en el análisis
        def calculate_risk(tool_wear, torque, rotational_speed):
            risk_score = 0
            
            # Tool wear risk
            if tool_wear > 200:
                risk_score += 3
            elif tool_wear > 150:
                risk_score += 2
            elif tool_wear > 100:
                risk_score += 1
            
            # Torque risk
            if torque > 60:
                risk_score += 3
            elif torque > 50:
                risk_score += 2
            elif torque > 40:
                risk_score += 1
            
            # Rotational speed risk
            if rotational_speed > 2500 or rotational_speed < 1400:
                risk_score += 2
            
            # Determinar nivel de riesgo
            if risk_score >= 5:
                return "🔴 ALTO RIESGO"
            elif risk_score >= 3:
                return "🟡 MEDIO RIESGO"
            else:
                return "🟢 BAJO RIESGO"
        
        # Ejemplos de evaluación
        test_cases = [
            (50, 35, 1600),   # Bajo riesgo
            (180, 55, 2200),  # Medio riesgo  
            (220, 65, 2800),  # Alto riesgo
        ]
        
        for i, (tool_wear, torque, speed) in enumerate(test_cases, 1):
            risk = calculate_risk(tool_wear, torque, speed)
            print(f"Caso {i}: Tool Wear={tool_wear}, Torque={torque}, Speed={speed} -> {risk}")
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        self.exploratory_analysis()
        self.visualize_data()
        self.train_models()
        self.feature_importance()
        self.risk_assessment_system()

# Ejecutar análisis
if __name__ == "__main__":
    # Asegúrate de que el archivo esté en la misma carpeta
    analyzer = PredictiveMaintenance('ai4i2020.csv')
    analyzer.run_complete_analysis()