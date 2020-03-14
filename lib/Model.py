from catboost import CatBoostClassifier
from Gesture import Gesture

class Model(object):
    def __init__(self, detector_window, classifier_window):
        self.detector_window = detector_window
        self.classifier_window = classifier_window
        self.detector = CatBoostClassifier()
        self.detector.load('./models/catboost/detector.pkl')
        self.classifier = CatBoostClassifier()
        self.classifier.load('./models/catboost/classifier.pkl')
        self.gesture_sequence = Gesture()

    def __call__(self, hand_landmarks):
        self.gesture_sequence.push(hand_landmarks)
        if len(self.gesture_sequence) > self.classifier_window:
            self.gesture_sequence.drop_first()
        tail_detect_vector = self.gesture_sequence.data(-self.detector_window)
        detector_predict = self.detector.predict(tail_detect_vector)
        if detector_predict:
            tail_classify_vector = self.gesture_sequence.data(-self.classifier_window)
            classifier_predict = self.classifier.predict(tail_classify_vector)
            if classifier_predict != 'No gesture':
                return classifier_predict
            else:
                return None
        else:
            return None
