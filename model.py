from classifier.mlp_classifier import *
import json
from classifier.svm_classifier import SVM

config = ai_path + '/configs/mlp_config.json'
config = json.load(open(config, 'r'))

svm_model = SVM()

cls = MLClassifier(config)
# cls.train(train_data, test_data)
# cls.train_multiclass(train_data, test_data)
cls.load_multi_class()


# print(cls.predict("Cho mình xin biểu mẫu cấp t24 với"))


def handle_request(message):
    if level1_filter(message) == 1:
        return level2_filter(message)
    return 'other'

def level1_filter(message):
    return svm_model.predict(message)
    

def level2_filter(message):
    label = cls.predict(message)
    return label


if __name__ == '__main__':
    # msg = 'em là thực tập sinh có được cấp t24 không'
    # print(handle_request(msg))
    msg = 'fuck you sb'
    print(handle_request(msg))
