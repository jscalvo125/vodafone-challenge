import pandas as pd
import spacy
import string


class ProductsRecommender:

    def __init__(self, train_file_name: str, user_id_column: str, product_id_col: str, model: str = "en_core_web_md"):
        # Create our list of punctuation marks
        self._train_dataset = None
        self._collapsed_df_by_user = None
        self._punctuations = string.punctuation
        self._nlp = spacy.load(model)
        try:
            self._train_dataset = pd.read_json(train_file_name, lines=True,
                                               orient='records')
            df_with_cols_of_interest = self._train_dataset[[user_id_column, product_id_col]]
            df_with_cols_of_interest = df_with_cols_of_interest.sort_values(by=user_id_column)
            self._collapsed_df_by_user = df_with_cols_of_interest.groupby([user_id_column],
                                                                    as_index=False).agg(' '.join).reset_index()
            self._collapsed_df_by_user = self._collapsed_df_by_user.rename(columns={product_id_col: "text"})
            # make documents
            docs = []
            for i in self._collapsed_df_by_user.index:
                if self._collapsed_df_by_user['text'][i] != '':
                    doc = self._nlp(self._collapsed_df_by_user['text'][i])
                    docs.append(doc)
            self._collapsed_df_by_user['doc'] = docs

        except Exception as inst:
            raise ValueError("I could not open the file {}".format(train_file_name))

    def recommend(self, user_id: str, user_id_column: str, product_id_col: str, top_n: int = 5):
        if user_id_column not in self._train_dataset.keys():
            raise AttributeError("user ID column not found")
        if product_id_col not in self._train_dataset.keys():
            raise AttributeError("Product ID column not found")
        # creating the 'text' attribute -> a merge of product_ids by user_id
        user_reviews = self._train_dataset[self._train_dataset[user_id_column] == user_id]
        user_text = ' '.join(user_reviews[product_id_col])
        top_users = self.calculate_top_users(self._collapsed_df_by_user, user_id_column, user_id, user_text, top_n)
        top_products = self.get_top_n_products(user_text, top_users, top_n)
        return top_products

    def calculate_top_users(self, df, user_id_column, user_id, text, n=10):
        list_sim = []
        doc_ref = self._nlp(text)
        scores = []
        for i in df.index:
            if df[user_id_column][i] == user_id:
                continue
            try:
                if df['text'][i] != '':
                    score = doc_ref.similarity(df['doc'][i])
                    scores.append(score)
                    list_sim.append((i, df['doc'][i], score))
            except:
                continue
        indexed_scores = pd.Series(scores)
        top_n = indexed_scores.nlargest(n)
        res = [tup for tup in list_sim if any(i in tup for i in top_n.index)]
        return res

    def get_top_n_products(self, user_text, top_users, top_n):
        top_prods = []
        user_products = user_text.split(' ')
        for similar_user in top_users:
            for product_token in similar_user[1]:
                if product_token.text not in user_products:
                    top_prods.append(product_token.text)
            if len(top_prods) >= top_n:
                return top_prods
        return top_prods
