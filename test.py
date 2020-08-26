from classifier.flair_intent_classifier import FlairDomainClassifier

config = 'D:\\flair_classification\\configs\\config.json'

cls = FlairDomainClassifier(config)

train_folder = 'D:\\flair_classification\\data'
cls._train(train_folder)