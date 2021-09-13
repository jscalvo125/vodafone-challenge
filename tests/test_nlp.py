import pickle

from spacy.scorer import Scorer

from nlp.nlp_training_module import NlpTrainer
from nlp.recommender import ProductsRecommender
import unittest
import os
from spacy.language import Language

dir_name = os.path.dirname(__file__)
train_data_name = os.path.join(dir_name, "../data/dataset_en_train_dev.json")
test_data_name = os.path.join(dir_name, "../data/dataset_en_test_dev.json")
test_prod_data_name = os.path.join(dir_name, "../data/dataset_en_test.json")
model_path_name = os.path.join(dir_name, "../models/en_core_web_lg.pkl")


class TestNlpTrainer(unittest.TestCase):

    def test_constructor(self):
        with self.assertRaises(ValueError) as ctx:
            nlp_trainer = NlpTrainer('wrongTrainFile', 'wrongTestFile')
        self.assertEqual("I could not open the file {}".format('wrongTrainFile'), str(ctx.exception))
        with self.assertRaises(ValueError) as ctx2:
            nlp_trainer = NlpTrainer(train_data_name, 'wrongTestFile')
        self.assertEqual("I could not open the file {}".format('wrongTestFile'), str(ctx2.exception))
        nlp_trainer = NlpTrainer(train_data_name, test_data_name)
        self.assertEqual(len(nlp_trainer._train_dataset), 100)
        self.assertEqual(len(nlp_trainer._test_dataset), 11)

    def test_train_model(self):
        nlp_trainer = NlpTrainer(train_data_name, test_data_name)
        with self.assertRaises(AttributeError) as column_ex:
            nlp_trainer.train_model(text_col='blah', label_col='product_category')
        self.assertEqual('text column {} not found in the train or test datasets'.format('blah'), str(column_ex.exception))
        with self.assertRaises(AttributeError) as column_ex2:
            nlp_trainer.train_model(text_col='review_body', label_col='blah')
        self.assertEqual('label column {} not found in the train or test datasets'.format('blah'), str(column_ex2.exception))
        results = nlp_trainer.train_model(text_col='review_body', label_col='product_category')
        self.assertIsNotNone(results)

    def test_evaluate(self):
        nlp_trainer = NlpTrainer(train_data_name, test_prod_data_name, model="en_core_web_lg")
        with open(model_path_name, "rb") as input_file:
            spacy_pipeline = pickle.load(input_file)
            nlp_trainer._nlp = spacy_pipeline
            categories = nlp_trainer._test_dataset['product_category'].unique()

            (train_texts, train_cats), (test_texts, test_cats) = nlp_trainer.load_data(nlp_trainer._train_dataset,
                                                                                nlp_trainer._test_dataset,
                                                                                text_col='review_body',
                                                                                label_col='product_category',
                                                                                categories=categories)
            # Processing the final format of training data
            gold_standard = nlp_trainer.transform_tuples_in_gold_standard(test_texts, nlp_trainer.normalize_categories(test_cats,
                                                                                                                       categories))
            result = spacy_pipeline.evaluate(gold_standard)
            print("Model results per category:")
            for cat_result in result.scores['textcats_per_cat']:
                print("{}: {}".format(cat_result, result.scores['textcats_per_cat'][cat_result]))
            self.assertEqual(31, len(result.scores['textcats_per_cat']))
            # print("Model's F1 is: {}".format(str(result)))
            # scores = result.score_cats(
            #     gold_standard,
            #     "cats",
            #     labels=categories
            # )
            # print("cats_micro_p {}".format(scores["cats_micro_p"]))
            # print("cats_micro_r {}".format(scores["cats_micro_r"]))
            # print("cats_micro_f {}".format(scores["cats_micro_f"]))
            # print("cats_macro_p {}".format(scores["cats_macro_p"]))
            # print("cats_macro_r {}".format(scores["cats_macro_r"]))
            # print("cats_macro_f {}".format(scores["cats_macro_f"]))
            # print("cats_macro_auc {}".format(scores["cats_macro_auc"]))

    def test_recommend(self):
        recommender = ProductsRecommender(test_data_name, 'reviewer_id', 'product_id')
        recommended_products = recommender.recommend('reviewer_en_0970351', 'reviewer_id', 'product_id')
        self.assertEqual(len(recommended_products), 5)
