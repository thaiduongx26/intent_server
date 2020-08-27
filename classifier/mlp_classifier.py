from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import numpy as np
import unidecode
import operator
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
import re, pickle
from tqdm import tqdm
from classifier import ai_path

x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))

stopwords = ['anh', 'em', 'minh', 'toi', 'tao', 'toa', 't24', 'coi', 'boi', 'ban', 'may', 'sasa', 'chao', 'hello', 'he',
             'nho', 'nhe', ]

remove_regex = [re.compile(f'\s+{w}') for w in stopwords] + [re.compile(r'\s+e\s+')]
edit_regex = {re.compile('cbnv'): 'nhan vien', re.compile('tk'): 'tai khoan',
              re.compile('user'): 'tai khoan', re.compile('dc'): 'duoc', }


def feature_analysis(data_dict):
    top_feature = {}
    for tag in data_dict:
        print("tag: ", tag)
        if tag == 'others': continue
        samples_tag = [unidecode.unidecode(x) for x in data_dict[tag]]
        other_samples = [unidecode.unidecode(x) for t in data_dict for x in data_dict[t] if t != tag and t != 'others']
        all_sample = samples_tag + other_samples
        vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords)
        vectorizer.fit(all_sample)
        vocab = vectorizer.vocabulary_
        big_tag_sample = '. '.join(samples_tag)
        big_other_sample = '. '.join(other_samples)
        tag_vec = vectorizer.transform([big_tag_sample]).todense().tolist()[0]
        other_vec = vectorizer.transform([big_other_sample]).todense().tolist()[0]
        index_to_vocab = {vocab[word]: word for word in vocab}
        feature_dict = {index_to_vocab[i]: tag_vec[i] / other_vec[i] if other_vec[i] != 0 else tag_vec[i] for i in
                        range(len(tag_vec))}
        top_feature[tag] = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)
        top_feature[tag] = [x[0] for x in top_feature[tag] if x[1] > 0]
    return top_feature


uncare_tags = ["Cấp mới T24_Không xác định ý định\nint_create_t24_need_confirm", 'others']
ignore_tags = ['Cấp mới T24_Không xác định ý định\nint_create_t24_need_confirm',
               'Cấp mới T24_Đối tượng\nint_create_t24_subject', 'Cấp mới T24_Thời gian cam kết', 'Cấp mới T24_FAQ',
               'others']


def preprocess(sentence):
    sentence = ' ' + sentence
    for reg in remove_regex:
        sentence = re.sub(reg, '', sentence)
    for reg in edit_regex:
        sentence = re.sub(reg, edit_regex[reg], sentence)
    # print("sentence: ", sentence)
    # 1/0
    return sentence


def analysis_error(text, labels, predictions):
    print("labels: ", labels)
    print("predictions: ", predictions)
    pass


class MLClassifier(object):
    def __init__(self, config):
        self.config = config
        self.config['vectorizer_path'] = ai_path + self.config['vectorizer_path']
        self.config['cls_path'] = ai_path + self.config['cls_path']
        self.vectorizer = None
        self.cls = None
        pass

    def train_for_a_tag(self, x_train, labels_train, x_test, labels_test, ngram=None, max_feature=None, tag=None,
                        get_result=False):
        vectorizers = CountVectorizer(ngram_range=ngram, max_features=max_feature)
        vectorizers.fit(x_train)
        x = vectorizers.transform(x_train)
        cls = MLPClassifier(max_iter=1000)
        cls.fit(x, labels_train)
        x = vectorizers.transform(x_test)
        y = cls.predict(x)
        f1 = f1_score(labels_test, y, average=None, labels=[tag])[0]
        if get_result:
            recall = recall_score(labels_test, y, average=None, labels=[tag])[0]
            precision = precision_score(labels_test, y, average=None, labels=[tag])[0]
            return [precision, recall, f1]
        return f1

    def train(self, train_data, test_data):
        top_feature = feature_analysis(data_dict=train_data)
        config = {}
        output = {}
        for tag in tqdm(train_data):
            if tag in ignore_tags: continue
            samples_tag = [preprocess(unidecode.unidecode(x.lower())) for x in train_data[tag]]
            # samples_tag = [preprocess(x.lower()) for x in train_data[tag]]
            other_samples = [preprocess(unidecode.unidecode(x.lower())) for t in train_data for x in train_data[t] if
                             t != tag and t not in uncare_tags]
            # other_samples = [preprocess(x.lower()) for t in train_data for x in train_data[t] if
            #                  t != tag and t not in uncare_tags]
            samples_tag_test = [preprocess(unidecode.unidecode(x.lower())) for x in test_data[tag]]
            other_samples_test = [preprocess(unidecode.unidecode(x.lower())) for t in test_data for x in test_data[t] if
                                  t != tag and t != 'others']
            labels_train = [tag] * len(samples_tag) + ['other'] * len(other_samples)
            x_train = samples_tag + other_samples
            labels_test = [tag] * len(samples_tag_test) + ['other'] * len(other_samples_test)
            x_test = samples_tag_test + other_samples_test
            f1_dict = {}
            for n_gram in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]:
                for max_feature in [50, 100, 150, 200, 250]:
                    f1 = self.train_for_a_tag(x_train, labels_train, x_test, labels_test, n_gram, max_feature, tag=tag,
                                              get_result=False)
                    f1_dict[f1] = (n_gram, max_feature)
            best_config = f1_dict[sorted(f1_dict, reverse=True)[0]]
            config[tag] = best_config
            result = self.train_for_a_tag(x_train, labels_train, x_test, labels_test, best_config[0], best_config[1],
                                          tag=tag, get_result=True)
            output[tag] = result
        print("final_result: ", output)

    def train_multiclass(self, train_data, test_data):
        X_train = []
        label_train = []
        x_test = []
        label_test = []

        def get_data_tag(data, tag):
            if tag in uncare_tags: return [], []
            samples_tag = [preprocess(unidecode.unidecode(x.lower())) for x in data[tag]]
            if tag in ignore_tags:
                labels_tag = ['others'] * len(samples_tag)
            else:
                labels_tag = [tag] * len(samples_tag)
            return samples_tag, labels_tag

        for tag in train_data:
            samples_tag, labels_tag = get_data_tag(train_data, tag)
            X_train.extend(samples_tag)
            label_train.extend(labels_tag)
        for tag in test_data:
            samples_tag, labels_tag = get_data_tag(test_data, tag)
            x_test.extend(samples_tag)
            label_test.extend(labels_tag)
        self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        x = self.vectorizer.fit_transform(X_train)
        self.cls = MLPClassifier(max_iter=1000)
        self.cls.fit(x, label_train)
        x = self.vectorizer.transform(x_test)
        y = self.cls.predict(x)
        print(classification_report(label_test, y))
        self.save_multi_class()

    def save_multi_class(self):
        with open(self.config['vectorizer_path'], 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(self.config['cls_path'], 'wb') as f:
            pickle.dump(self.cls, f)

    def load_multi_class(self):
        with open(self.config['vectorizer_path'], 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.config['cls_path'], 'rb') as f:
            self.cls = pickle.load(f)

    def predict(self, sentence):
        if len(preprocess(unidecode.unidecode(sentence.lower())).strip()) == 0:
            return 'other'
        sentence = [preprocess(unidecode.unidecode(sentence.lower()))]
        x = self.vectorizer.transform(sentence)
        if sum(x.todense().tolist()[0]) == 0:
            return 'other'
        label = self.cls.predict(x)[0]
        return label


if __name__ == '__main__':
    import json

    with open("D:\\intent_server\\data\\train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open("D:\\intent_server\\data\\test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    config = ai_path + '/configs/mlp_config.json'
    config = json.load(open(config, 'r'))

    cls = MLClassifier(config)
    # cls.train(train_data, test_data)
    cls.train_multiclass(train_data, test_data)
    cls.load_multi_class()
    print(cls.predict("Cho mình xin biểu mẫu cấp t24 với"))
    # feature_analysis(data)
