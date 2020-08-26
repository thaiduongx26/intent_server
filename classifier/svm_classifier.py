import pickle
import os
from configs.svm_config import Config_SVM

class SVM():
    def __init__(self):
        config = Config_SVM()
        self.vectorizer = pickle.load(open(config.VECTORIZER_PATH, 'rb'))
        self.model = pickle.load(open(config.MODEL_PATH, 'rb'))
    
    def predict(self, text):
        vector = self.vectorizer.transform([text])
        return self.model.predict(vector)[0]