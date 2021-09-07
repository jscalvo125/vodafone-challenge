from nlp.nlp_training_module import NlpTrainer
from nlp.recommender import ProductsRecommender
import unittest
import os

dirname = os.path.dirname(__file__)
train_data_name = os.path.join(dirname, "../data/dataset_en_train_dev.json")
test_data_name = os.path.join(dirname, "../data/dataset_en_test_dev.json")


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
        self.assertEqual(len(nlp_trainer._test_dataset), 10)

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

    def test_recommend(self):
        recommender = ProductsRecommender(test_data_name, 'reviewer_id', 'product_id')
        recommended_products = recommender.recommend('reviewer_en_0970351', 'reviewer_id', 'product_id')
        self.assertEqual(len(recommended_products), 5)
