
import os
import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class RetinalAnalyzer:
    """
    Educational retinal analysis prototype.
    No independently verified clinical accuracy is claimed.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.diseases = [
            'أمراض القلب التاجية',
            'ارتفاع ضغط الدم',
            'أمراض الكلى',
            'السكري',
            'تصلب الشرايين',
            'أمراض الأوعية الدموية',
            'اعتلال الشبكية',
            'جلوكوما'
        ]

    def build_retinal_model(self):
        """بناء نموذج تحليل شبكية العين باستخدام GradientBoosting"""
        # إنشاء نموذج منفصل لكل مرض (نهج متعدد التصنيفات)
        models = {}
        for disease in self.diseases:
            models[disease] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

        self.model = models
        return models

    def analyze_retina(self, eye_image_path):
        """Educational scores only; reject missing or untrained models."""
        from sklearn.utils.validation import check_is_fitted
        if self.model is None or not getattr(self, 'trained_diseases', set()):
            raise RuntimeError("No trained retinal classifiers are available")
        image = cv2.imread(str(eye_image_path))
        if image is None:
            raise ValueError("Could not read retinal image")
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        features = self.scaler.transform([self.extract_features(image)])
        results = {}
        for disease, model in self.model.items():
            if disease not in self.trained_diseases:
                results[disease] = {'probability': None, 'risk_level': 'Unavailable', 'recommended_action': 'Model not trained'}
                continue
            check_is_fitted(model)
            positive = list(model.classes_).index(1)
            probability = float(model.predict_proba(features)[0][positive])
            results[disease] = {
                'probability': probability,
                'risk_level': 'Research score only',
                'recommended_action': 'Not a clinical assessment; consult a qualified clinician.'
            }
        return results

    def extract_features(self, image):
        """استخراج الميزات من صورة شبكية العين"""
        # تحويل الصورة إلى مصفوفة مسطحة
        flattened = image.flatten() / 255.0
        
        # إضافة ميزات إضافية مثل متوسط القيم، الانحراف المعياري، إلخ
        mean_rgb = np.mean(image, axis=(0, 1))
        std_rgb = np.std(image, axis=(0, 1))
        
        # دمج الميزات
        features = np.concatenate([
            mean_rgb, std_rgb,
            # يمكن إضافة المزيد من الميزات المستخرجة هنا
            # مثل: تحليل الحواف، تحليل النسيج، إلخ
        ])
        
        return features

    def get_risk_level(self, probability):
        """تحديد مستوى الخطر بناءً على الاحتمالية"""
        if probability > 0.8:
            return "خطر عالي"
        elif probability > 0.6:
            return "خطر متوسط"
        elif probability > 0.3:
            return "خطر منخفض"
        else:
            return "طبيعي"

    def get_recommendation(self, disease, probability):
        """تقديم التوصيات الطبية"""
        if probability > 0.8:
            return f"يُنصح بزيارة طبيب مختص في {disease} فوراً"
        elif probability > 0.6:
            return f"يُنصح بإجراء فحوصات إضافية لـ {disease}"
        elif probability > 0.3:
            return f"متابعة دورية لحالة {disease}"
        else:
            return "لا توجد مؤشرات خطر حالياً"
    
    def evaluate_model(self, X_test, y_test_dict):
        """تقييم نماذج تحليل شبكية العين
        
        المعلمات:
        ----------
        X_test : مصفوفة الميزات للاختبار
        y_test_dict : قاموس يحتوي على تصنيفات لكل مرض
        
        العائد:
        -------
        dict : قاموس يحتوي على نتائج التقييم لكل مرض
        """
        if self.model is None:
            raise ValueError("يجب بناء وتدريب النماذج أولاً")
            
        # تطبيع البيانات
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        for disease in self.diseases:
            if disease in y_test_dict and disease in getattr(self, 'trained_diseases', set()):
                # التنبؤ باستخدام النموذج
                y_pred = self.model[disease].predict(X_test_scaled)
                y_prob = self.model[disease].predict_proba(X_test_scaled)[:, list(self.model[disease].classes_).index(1)]
                
                # حساب مقاييس الأداء
                accuracy = accuracy_score(y_test_dict[disease], y_pred)
                report = classification_report(y_test_dict[disease], y_pred, output_dict=True)
                
                results[disease] = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1-score': report['weighted avg']['f1-score'],
                    'predictions': y_pred.tolist(),
                    'probabilities': y_prob.tolist()
                }
            else:
                results[disease] = {'error': f'لا توجد بيانات اختبار لـ {disease}'}
                
        return results
            
    def train_models(self, X_train, y_train_dict, X_val=None, y_val_dict=None):
        """تدريب نماذج تحليل شبكية العين
        
        المعلمات:
        ----------
        X_train : مصفوفة الميزات للتدريب
        y_train_dict : قاموس يحتوي على تصنيفات لكل مرض
        X_val : مصفوفة الميزات للتحقق (اختياري)
        y_val_dict : قاموس يحتوي على تصنيفات التحقق لكل مرض (اختياري)
        
        العائد:
        -------
        dict : قاموس يحتوي على نتائج التدريب لكل مرض
        """
        if self.model is None:
            self.build_retinal_model()
            
        # تطبيع البيانات
        self.trained_diseases = set()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {}
        for disease in self.diseases:
            if disease in y_train_dict:
                # تدريب النموذج لهذا المرض
                if set(np.unique(y_train_dict[disease])) != {0, 1}:
                    raise ValueError('Expected binary labels 0 and 1')
                self.model[disease].fit(X_train_scaled, y_train_dict[disease])
                self.trained_diseases.add(disease)
                
                # تقييم النموذج إذا كانت بيانات التحقق متوفرة
                if X_val is not None and y_val_dict is not None and disease in y_val_dict:
                    X_val_scaled = self.scaler.transform(X_val)
                    y_pred = self.model[disease].predict(X_val_scaled)
                    accuracy = accuracy_score(y_val_dict[disease], y_pred)
                    results[disease] = {'accuracy': accuracy}
                else:
                    # تقييم على بيانات التدريب
                    y_pred = self.model[disease].predict(X_train_scaled)
                    accuracy = accuracy_score(y_train_dict[disease], y_pred)
                    results[disease] = {'accuracy': accuracy, 'note': 'تقييم على بيانات التدريب'}
            else:
                results[disease] = {'error': f'لا توجد بيانات تدريب لـ {disease}'}
                
        return results

if __name__ == "__main__":
    print("Supply authorized labeled retinal data and trained classifiers for this educational experiment.")
