import logging
import random
from tqdm import tqdm
import pandas as pd
import spacy
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
import string
# python -m spacy download en_core_web_sm and lagre tooo to make en_core_web_sm works properly


class NlpTrainer:

    def __init__(self, train_file_name: str, test_file_name: str, model: str = "en_core_web_sm"):
        # Create our list of punctuation marks
        self._train_dataset = None
        self._test_dataset = None
        self._punctuations = string.punctuation
        self._nlp = spacy.load(model)
        try:
            self._train_dataset = pd.read_json(train_file_name, lines=True,
                                               orient='records')
        except Exception as inst:
            raise ValueError("I could not open the file {}".format(train_file_name))
        else:
            try:
                self._test_dataset = pd.read_json(test_file_name, lines=True,
                                                  orient='records')
            except Exception as inst:
                raise ValueError("I could not open the file {}".format(test_file_name))

    # # Creating our tokenizer function
    # def spacy_tokenizer(self, sentence: str):
    #     # Creating our token object, which is used to create documents with linguistic annotations.
    #     tokens = self._parser(sentence)
    #     # Lemmatizing each token and converting each token into lowercase
    #     tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    #     # Removing stop words
    #     tokens = [word for word in tokens if word not in self._stop_words and word not in self._punctuations]
    #     # return preprocessed list of tokens
    #     return tokens

    def gather_statistics(self):
        logging.warning(self._train_dataset.head())
        logging.warning('Dataset shape {}'.format(self._train_dataset.shape))
        logging.warning('Dataset information:')
        logging.warning(self._train_dataset.info())

    def make_cats(self, y,  categories):
        new_cats = {}
        for category in categories:
            if y == category:
                new_cats[category] = 1.0
        #        else:
        #            new_cats[category] = 0.0
        return new_cats

    def evaluate_test_set(self, text_gold_ds):
        return self._nlp.evaluate(text_gold_ds)

    def batch_evaluate(self, texts, cats):
        docs = (self._nlp(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, doc in enumerate(docs):
            gold = cats[i]
            max_score = 0.0
            best_label = ''
            for label, score in doc.cats.items():
                if max_score < score:
                    max_score = score
                    best_label = label

            # print("document {}".format(str(i)))
            # print("predicted label {} and score {}".format(best_label, max_score))
            # print("real label {}".format(gold))
            for label, score in doc.cats.items():
                if score < max_score and label not in gold:
                    tn += 1
                elif score < max_score and gold[label] >= 0.5:
                    fn += 1

            if best_label not in gold:
                fp += 1
            elif gold[best_label] >= 0.5:
                tp += 1.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"pipeline":self._nlp, "textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

    def tranform_dataframe_into_gold_standard(self, data_frame, text_col: str, label_col: str):
        gold_ds = []
        for idx, instance in tqdm(data_frame.iterrows()):
            doc = self._nlp(instance[text_col])
            #    print([token.text for token in doc])
            gold_annotation = GoldParse(doc, cats={instance[label_col]: 1.0})
            gold_ds.append(gold_annotation)
        return gold_ds

    def transform_tuples_in_gold_standard(self, gs_texts, gs_categories):
        return list(zip(gs_texts, [{'cats': cats} for cats in gs_categories]))

    def load_data(self, train_dataframe, test_dataframe, text_col: str, label_col: str, categories):
        # Converting the dataframe into a list of tuples
        self._train_dataset['cat_tuples'] = self._train_dataset.apply(lambda row: (
                                                 row[text_col], row[label_col].encode().decode("unicode-escape")),
                                                 axis=1)
        self._test_dataset['cat_tuples'] = self._test_dataset.apply(lambda row: (
                                                 row[text_col], row[label_col].encode().decode("unicode-escape")),
                                                 axis=1)
        train_data = train_dataframe['cat_tuples'].tolist()
        test_data = test_dataframe['cat_tuples'].tolist()
        texts, labels = zip(*train_data)
        test_texts, test_labels = zip(*test_data)
        # get the categories for each review
        cats = []
        for y in labels:
            cats.append(self.make_cats(y, categories))

        test_cats = []
        for y in test_labels:
            test_cats.append(self.make_cats(y, categories))
        # Splitting the training and evaluation data
        return (texts, cats), (test_texts, test_cats)

    def normalize_categories(self, text_cats, categories):
        new_text_cats = []
        for text_cat in text_cats:
            new_text_cat = {}
            for cat in text_cat.keys():
                new_text_cat[cat] = text_cat[cat]
            for cat in categories:
                if cat not in new_text_cat.keys():
                    new_text_cat[cat] = 0.0
            new_text_cats.append(new_text_cat)
        return new_text_cats

    def train_model(self, text_col: str, label_col: str, classifier="simple_cnn"):
        logging.warning("Validating columns")
        if text_col not in self._train_dataset.keys() or text_col not in self._test_dataset.keys():
            raise AttributeError("text column {} not found in the train or test datasets".format(text_col))
        if label_col not in self._train_dataset.keys() or label_col not in self._test_dataset.keys():
            raise AttributeError("label column {} not found in the train or test datasets".format(label_col))
        logging.warning("Transforming datasets into spacy-like format")

        text_cat = self._nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": classifier})
        self._nlp.add_pipe(text_cat, last=True)

        categories = self._train_dataset[label_col].unique()
        for category in categories:
            text_cat.add_label(category)

        # Calling the load_data() function
        (train_texts, train_cats), (test_texts, test_cats) = self.load_data(self._train_dataset, self._test_dataset,
                                                                            text_col, label_col,
                                                                            categories)
        # Processing the final format of training data
        train_data = self.transform_tuples_in_gold_standard(train_texts, self.normalize_categories(train_cats,
                                                                                                   categories))
        # print(train_data[:2])
        # print(test_data[:2])
        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self._nlp.pipe_names if pipe not in pipe_exceptions]
        # training time
        with self._nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self._nlp.begin_training()
            logging.warning("Training the model...")
            logging.warning("Iterations {}".format(3))
            # logging.warning("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            for i in tqdm(range(3)):
                losses = {}
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                batches = minibatch(train_data, size=1)
                print("batch size {}".format(len(train_data)))
                for batch in batches:
                    # print("batch {}".format(batch))
                    texts, annotations = zip(*batch)
                    doc = self._nlp(texts[0])
                    # tp, fp, tn, fn = self.update_metrics(doc, annotations[0], tp, fp, tn, fn)
                    self._nlp.update(texts, annotations, sgd=optimizer, drop=0.1, losses=losses)
        return self.batch_evaluate(test_texts, test_cats)
#        return self.evaluate_test_set(test_data)

        # vectorizer = TfidfVectorizer(tokenizer=self.spacy_tokenizer)
        #
        # if feature_type == "tf-idf":
        #     vectorizer = TfidfVectorizer(tokenizer=self.spacy_tokenizer)
        # elif feature_type == "bow":
        #     vectorizer = CountVectorizer(tokenizer=self.spacy_tokenizer, ngram_range=(1, 2))  # I want bi-grams too
        # elif feature_type == "embedding":
        #     pass
        # else:
        #     raise AttributeError('wrong feature type. Use tf-idf, bow or embedding')
        # pipe = Pipeline([("cleaner", Predictors()),
        #                  ('vectorizer', vectorizer),
        #                  ('classifier', classifier)])
        # pipe.fit(X_train, Y_train)

