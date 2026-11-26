
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class MedicalImageAnalyzer:
    """
    Educational medical image analysis prototype.
    No independently verified clinical accuracy is claimed.
    """

    def __init__(self):
        self.model = None
        self.input_size = 224 * 224 * 3  # Flattened image size
        self.num_classes = 10  # عدد أنواع الأمراض المختلفة
        self.scaler = StandardScaler()
        self.disease_labels = [
            'سرطان الرئة', 'سرطان الثدي', 'سرطان الجلد', 
            'أمراض القلب', 'السكتة الدماغية', 'التهاب المفاصل',
            'أمراض الكلى', 'أمراض الكبد', 'طبيعي', 'غير محدد'
        ]

    def build_model(self):
        """بناء نموذج تصنيف الصور الطبية باستخدام RandomForest"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model = model
        return model

    def preprocess_image(self, image_path):
        """Read an image or fail explicitly; never infer from a fabricated zero array."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read image")
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        features = np.asarray(image, dtype=np.float32).reshape(1, -1) / 255.0
        return self.scaler.transform(features) if getattr(self, 'scaler_fitted', False) else features

    def _label(self, value):
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(self.disease_labels):
            return self.disease_labels[int(value)]
        if isinstance(value, str) and value in self.disease_labels:
            return value
        raise ValueError("Unsupported class label")

    def predict_disease(self, image_path, confidence_threshold=0.85):
        """Return educational model scores only after successful training."""
        from sklearn.utils.validation import check_is_fitted
        if self.model is None or not getattr(self, 'scaler_fitted', False):
            raise RuntimeError("A trained model and fitted preprocessing are required")
        check_is_fitted(self.model)
        probabilities = self.model.predict_proba(self.preprocess_image(image_path))[0]
        labels = [self._label(label) for label in self.model.classes_]
        if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid classifier output")
        predicted = int(np.argmax(probabilities))
        return {
            'disease': labels[predicted],
            'confidence': float(probabilities[predicted]),
            # Model confidence is not clinical reliability.
            'reliable': False,
            'all_probabilities': dict(zip(labels, map(float, probabilities))),
            'disclaimer': 'Educational classifier output; not a diagnosis.'
        }

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """تدريب النموذج على البيانات الطبية"""
        # تطبيع البيانات
        X_train_scaled = self.scaler.fit_transform(X_train)
        if self.model is None:
            self.build_model()
        self.scaler_fitted = False
        for label in np.unique(y_train):
            self._label(label)
        self.model.fit(X_train_scaled, y_train)
        self.scaler_fitted = True
        
        # تقييم النموذج إذا كانت بيانات التحقق متوفرة
        if X_val is not None and y_val is not None:
            return self.evaluate_model(X_val, y_val)
        
        return {'accuracy': self.model.score(X_train_scaled, y_train)}
        
    def evaluate_model(self, X_test, y_test):
        """تقييم أداء النموذج"""
        if self.model is None:
            return {'accuracy': 0.0, 'error': 'النموذج غير مدرب'}
            
        # تطبيع البيانات
        X_test_scaled = self.scaler.transform(X_test)
        
        # التنبؤ
        y_pred = self.model.predict(X_test_scaled)
        
        # حساب الدقة
        accuracy = accuracy_score(y_test, y_pred)
        
        # تقرير التصنيف
        report = classification_report(y_test, y_pred, labels=self.model.classes_, target_names=[self._label(label) for label in self.model.classes_], output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
             'classification_report': report
         }

if __name__ == "__main__":
    print("Supply authorized labeled training data and a non-sensitive sample image before running this educational classifier.")
