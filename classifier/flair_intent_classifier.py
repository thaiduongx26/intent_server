from rasa.flair.models import TextClassifier
from rasa.flair.data import TaggedCorpus, MultiCorpus
from rasa.flair.data_fetcher import NLPTaskDataFetcher
from rasa.flair.embeddings import *
from classifier import ai_path
import os, json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_config(config_path: str = None) -> Dict:
    if not config_path:
        config = {}
    elif not os.path.isfile(config_path):
        print("Your config path is not correct")
        config = {}
    else:
        with open(config_path, "rb") as f:
            config = json.load(f)
    return config

class FlairDomainClassifier(object):
    def __init__(self, config):
        if isinstance(config, str):
            self.config = load_config(config_path=config)
        else:
            self.config = config

    def _load(self):
        try:
            model_file = ai_path + '/' + self.config.get('model_dir') + '/' + self.config.get("model_file")
            if not os.path.isfile(model_file):
                raise ("Model is not exist")
            self.classifier = TextClassifier.load_from_file(model_file)
        except:
            import requests
            sess = requests.Session()
            if 'api' in self.config:
                url = self.config['api']
                log.info('call api')

                def call_api(sentences_processed: List = None):
                    if sentences_processed is None:
                        return []
                    r = sess.post(url, json=sentences_processed).json()
                    return r

                self._predict = call_api

    def _train(self, training_data) -> None:
        if isinstance(training_data, str):
            training_folder = training_data
        else:
            training_folder = training_data.convert_to_conll()
        cls_corpus = NLPTaskDataFetcher.load_classification_corpus(training_folder, test_file='test.csv',
                                                                   train_file='train.csv')
        corpus = MultiCorpus([cls_corpus])
        lm_fw = self.config['lm_fw']
        lm_bw = self.config['lm_bw']
        if not (os.path.isfile(lm_bw) and os.path.isfile(lm_fw)):
            print('U need put pretrained Flair add ' + lm_bw + ' for backward and ' + lm_fw + ' for foward')
        embedding_types: List[TokenEmbeddings] = [  #
            FlairEmbeddings(lm_fw),
            FlairEmbeddings(lm_bw),
            BytePairEmbeddings('vi', dim=100),
        ]
        document_embeddings = DocumentLSTMEmbeddings(embedding_types, hidden_size=512, reproject_words=True,
                                                     reproject_words_dimension=512)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False, attention=False)
        from rasa.flair.trainers import ModelTrainer
        if os.path.isfile(self.config['model_dir']):
            self.trainer = ModelTrainer.load_from_checkpoint(self.config['model_dir'], 'TextClassifier', corpus)
        else:
            self.trainer: ModelTrainer = ModelTrainer(classifier, corpus)
        self.trainer.train(base_path=self.config['model_dir'],
                           # save_path_sagemaker=self.config['save_path_sagemaker'],
                           learning_rate=0.1,
                           mini_batch_size=32,
                           max_epochs=100,
                           train_with_dev=True,
                           # save_checkpoint_to_s3=True,
                           # s3_path_for_saving_checkpoint=self.config['s3_path_for_saving_checkpoint'],
                           param_selection_mode=False,
                           checkpoint=True,
                           monitor_train=True)

    def _get_all_label(self):
        all_labels = [str(s) for s in self.classifier._get_all_labels()]
        return all_labels

    def _predict(self, sentences_processed: List = None):
        sentences_processed = [sentence if len(sentence.strip()) > 0 else '.' for sentence in sentences_processed]
        sentences = [Sentence(sentence) for sentence in sentences_processed]
        self.classifier.predict(sentences)
        return [{"name": sentence.labels[0].value, "confidence": sentence.labels[0].score} for sentence in sentences]
