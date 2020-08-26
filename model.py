from classifier.mlp_classifier import *
import json
config = ai_path + '/configs/mlp_config.json'
config = json.load(open(config, 'r'))

cls = MLClassifier(config)
# cls.train(train_data, test_data)
# cls.train_multiclass(train_data, test_data)
cls.load_multi_class()
# print(cls.predict("Cho mình xin biểu mẫu cấp t24 với"))


def handle_request(message):
    label = cls.predict(message)
    return label

if __name__ == '__main__':
    msg = 'em là thực tập sinh có được cấp t24 không'
    print(handle_request(msg))