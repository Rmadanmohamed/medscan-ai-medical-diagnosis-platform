import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from medical_image_analyzer import MedicalImageAnalyzer
from retinal_analyzer import RetinalAnalyzer
from disease_prediction import DiseasePredictionSystem

class MedicalPrototypeTests(unittest.TestCase):
    def test_untrained_and_unreadable_images_fail_closed(self):
        analyzer = MedicalImageAnalyzer()
        analyzer.build_model()
        with self.assertRaises(RuntimeError):
            analyzer.predict_disease('missing.png')
        with self.assertRaises(ValueError):
            analyzer.preprocess_image('missing.png')

    def test_class_labels_follow_fitted_model_not_probability_position(self):
        analyzer = MedicalImageAnalyzer()
        analyzer.model = DummyClassifier(strategy='prior')
        x = np.zeros((4, analyzer.input_size), dtype=np.float32)
        analyzer.train_model(x, np.array([3,3,3,8]))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'sample.png'
            self.assertTrue(cv2.imwrite(str(image), np.zeros((4,4,3), dtype=np.uint8)))
            result = analyzer.predict_disease(image)
        self.assertEqual(result['disease'], analyzer.disease_labels[3])
        self.assertEqual(set(result['all_probabilities']), {analyzer.disease_labels[3], analyzer.disease_labels[8]})
        self.assertFalse(result['reliable'])

    def test_retinal_preprocessing_matches_training_scaler(self):
        analyzer = RetinalAnalyzer()
        analyzer.build_retinal_model()
        with self.assertRaises(RuntimeError):
            analyzer.analyze_retina('missing.png')
        disease = analyzer.diseases[0]
        analyzer.model[disease] = DummyClassifier(strategy='prior')
        X = np.array([[0,0,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]], dtype=float)
        analyzer.train_models(X, {disease:np.array([0,1,0,1])})
        scaler = analyzer.scaler
        wrapped = Mock(wraps=scaler.transform)
        analyzer.scaler.transform = wrapped
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'sample.png'
            self.assertTrue(cv2.imwrite(str(image), np.zeros((4,4,3), dtype=np.uint8)))
            result = analyzer.analyze_retina(image)
        wrapped.assert_called_once()
        self.assertIsNone(result[analyzer.diseases[1]]['probability'])
        self.assertEqual(result[disease]['risk_level'], 'Research score only')

    def test_no_prognosis_or_predictions_without_training(self):
        system = DiseasePredictionSystem()
        system.build_prediction_models()
        self.assertIsNone(system.estimate_time_to_onset('cancer', .99))
        with self.assertRaises(RuntimeError):
            system.predict_disease_risk(pd.DataFrame())

if __name__ == '__main__':
    unittest.main()
