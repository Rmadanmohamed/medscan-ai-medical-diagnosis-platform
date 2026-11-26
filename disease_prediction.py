
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DiseasePredictionSystem:
    """
    نظام التنبؤ المبكر بالأمراض
    Educational disease-risk prototype.
    No independently verified clinical accuracy is claimed.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []

    def prepare_health_data(self, data):
        """إعداد البيانات الصحية للتحليل"""
        # البيانات الأساسية
        health_features = [
            'age', 'gender', 'bmi', 'blood_pressure_systolic', 
            'blood_pressure_diastolic', 'heart_rate', 'cholesterol',
            'glucose_level', 'family_history_heart', 'family_history_diabetes',
            'family_history_cancer', 'smoking', 'alcohol_consumption',
            'exercise_frequency', 'stress_level', 'sleep_hours'
        ]

        # البيانات المختبرية
        lab_features = [
            'white_blood_cells', 'red_blood_cells', 'hemoglobin',
            'platelets', 'creatinine', 'urea', 'alt', 'ast',
            'triglycerides', 'hdl_cholesterol', 'ldl_cholesterol'
        ]

        self.feature_names = health_features + lab_features
        return data[self.feature_names]

    def build_prediction_models(self):
        """بناء نماذج التنبؤ المختلفة للأمراض المتنوعة"""

        # نموذج أمراض القلب
        self.models['heart_disease'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        # نموذج السكري
        self.models['diabetes'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )

        # نموذج السرطان
        self.models['cancer'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42
        )

        # نموذج أمراض الكلى
        self.models['kidney_disease'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.15,
            random_state=42
        )

        return self.models

    def train_models(self, data, target_columns):
        """تدريب جميع نماذج التنبؤ"""
        self.scalers = {}
        X = self.prepare_health_data(data)

        for disease, model in self.models.items():
            if disease in target_columns:
                y = data[disease]
                if set(y.unique()) != {0, 1}:
                    raise ValueError(f'{disease} requires binary labels 0 and 1')

                # تقسيم البيانات
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # تطبيع البيانات
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # تدريب النموذج
                model.fit(X_train_scaled, y_train)

                # تقييم الأداء
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"دقة نموذج {disease}: {accuracy:.2%}")

                # حفظ المقياس
                self.scalers[disease] = scaler

    def predict_disease_risk(self, patient_data):
        """التنبؤ بمخاطر الأمراض لمريض معين"""
        predictions = {}

        # تحضير بيانات المريض
        if not self.scalers:
            raise RuntimeError('No trained disease classifiers are available')
        if len(patient_data) != 1:
            raise ValueError('Provide exactly one record for prediction')
        patient_features = self.prepare_health_data(patient_data)

        for disease, model in self.models.items():
            if disease in self.scalers:
                # تطبيع البيانات
                scaled_data = self.scalers[disease].transform(patient_features)

                # التنبؤ
                risk_probability = model.predict_proba(scaled_data)[0][list(model.classes_).index(1)]

                predictions[disease] = {
                    'risk_probability': float(risk_probability),
                    'risk_category': 'Research score only',
                    'time_to_onset': None,
                    'prevention_tips': [],
                    'disclaimer': 'Unvalidated educational score, not a diagnosis or prognosis.'
                }

        return predictions

    def categorize_risk(self, probability):
        """تصنيف مستوى المخاطر"""
        if probability > 0.8:
            return "خطر مرتفع جداً"
        elif probability > 0.6:
            return "خطر مرتفع"
        elif probability > 0.4:
            return "خطر متوسط"
        elif probability > 0.2:
            return "خطر منخفض"
        else:
            return "خطر منخفض جداً"

    def estimate_time_to_onset(self, disease, probability):
        """The classifiers are not survival models and cannot estimate onset."""
        return None

    def get_prevention_tips(self, disease, probability):
        """تقديم نصائح الوقاية المخصصة"""
        tips = {
            'heart_disease': [
                "ممارسة الرياضة 30 دقيقة يومياً",
                "تقليل الملح والدهون المشبعة",
                "الإقلاع عن التدخين",
                "إدارة التوتر والضغط النفسي"
            ],
            'diabetes': [
                "الحفاظ على وزن صحي",
                "تناول غذاء متوازن قليل السكر",
                "ممارسة الرياضة بانتظام",
                "مراقبة مستوى السكر دورياً"
            ],
            'cancer': [
                "تجنب التدخين والكحول",
                "تناول الخضار والفواكه",
                "حماية من أشعة الشمس",
                "إجراء فحوصات دورية"
            ],
            'kidney_disease': [
                "شرب كمية كافية من الماء",
                "تقليل الملح والبروتين",
                "مراقبة ضغط الدم",
                "تجنب الأدوية الضارة بالكلى"
            ]
        }

        return tips.get(disease, ["اتباع نمط حياة صحي"])

if __name__ == "__main__":
    print("Supply authorized labeled health data to train the educational classifiers before prediction.")
