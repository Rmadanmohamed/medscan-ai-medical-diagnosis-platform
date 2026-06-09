
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
        """معالجة الصورة الطبية قبل التحليل"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
                
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # استخراج الميزات البسيطة من الصورة
            image_flattened = image.flatten() / 255.0
            
            # تطبيع البيانات
            if self.model is not None and hasattr(self, 'scaler_fitted') and self.scaler_fitted:
                image_scaled = self.scaler.transform([image_flattened])
            else:
                # في حالة عدم تدريب النموذج بعد، نستخدم البيانات كما هي
                image_scaled = np.array([image_flattened])
                
            return image_scaled
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # إرجاع مصفوفة صفرية في حالة الخطأ
            return np.zeros((1, self.input_size))

    def predict_disease(self, image_path, confidence_threshold=0.85):
        """تشخيص المرض من الصورة الطبية"""
        if self.model is None:
            return {
                'disease': 'غير محدد',
                'confidence': 0.0,
                'reliable': False,
                'all_probabilities': {label: 0.0 for label in self.disease_labels}
            }
            
        processed_image = self.preprocess_image(image_path)
        
        # الحصول على احتمالات التصنيف
        probabilities = self.model.predict_proba(processed_image)[0]
        predicted_class = np.argmax(probabilities)
        max_confidence = probabilities[predicted_class]

        result = {
            'disease': self.disease_labels[predicted_class],
            'confidence': float(max_confidence),
            'reliable': max_confidence > confidence_threshold,
            'all_probabilities': {
                self.disease_labels[i]: float(probabilities[i]) 
                for i in range(len(self.disease_labels))
            }
        }

        return result

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """تدريب النموذج على البيانات الطبية"""
        # تطبيع البيانات
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.scaler_fitted = True
        
        # تدريب النموذج
        self.model.fit(X_train_scaled, y_train)
        
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
        report = classification_report(y_test, y_pred, target_names=self.disease_labels, output_dict=True)
        
        return {
            'accuracy': accuracy,
             'classification_report': report
         }

# مثال على الاستخدام
if __name__ == "__main__":
    analyzer = MedicalImageAnalyzer()
    model = analyzer.build_model()

    # تشخيص صورة طبية
    result = analyzer.predict_disease('medical_scan.jpg')
    print(f"التشخيص: {result['disease']}")
    print(f"مستوى الثقة: {result['confidence']:.2%}")
