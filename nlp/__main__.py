import logging
import os
from nlp import nlp_training_module
import pickle
import sys


if __name__ == '__main__':
    try:
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

        if "-f" in opts:
            print(" ".join(arg for arg in args))

        logging.info(f"Arguments count: {len(sys.argv)}")
        train_file_name = args[0]
        test_file_name = args[1]

        # train the model
        results = []
        for clf, name in (
                ("en_core_web_sm", "BoW and simple CNN"),
                ("en_core_web_md", "Word2Vec and simple CNN")):
            logging.info('=' * 80)
            logging.info(name)
            nlp_trainer = nlp_training_module.NlpTrainer(train_file_name, test_file_name, model=clf)
            results.append(nlp_trainer.train_model(text_col='review_body', label_col='product_category'))

        best_score = 0.0
        best_clf = results[0]['textcat_f']
        for result in results:
            if result['textcat_f'] > best_score:
                best_clf = result['pipeline']
                best_score = result['textcat_f']

        logging.warning("Saving the best  model which is {}".format(str(best_clf)))
        logging.warning("Model Score F-1 : {}".format(str(best_score)))

        dir_name = os.path.dirname(__file__)
        model_path_name = os.path.join(dir_name, "../models/trained_model.pkl")
        with open(model_path_name, 'wb') as handle:
            pickle.dump(best_clf, handle)
            logging.warning("Model saved into {} :)".format(model_path_name))

    except IndexError:
        raise SystemExit(f"Usage: python -m nlp -f <train_data_path_name.json> <test_data_path_name.json>")
