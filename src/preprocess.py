import pandas as pd
import unicodedata
import re

if __name__ == '__main__':
    ROOT = 'shopee_sentiment_data_set'
    data = pd.read_csv(f'{ROOT}/train.csv')

    reviews = data['text']
    p_reviews = []
    
    for i, r in enumerate(reviews):
        r = unicodedata.normalize('NFC', r)
        r = re.sub(r'^\w+', ' ', r)

    for review in reviews:
        review = review.lower()
        review = re.sub(r'_', ' ', review)
        review = re.sub(r'\W', ' ', review)
        review = re.sub(r'\s+', ' ', review)
        p_reviews.append(review)

    data['text'] = p_reviews
    data.to_csv(f'{ROOT}/p_train.csv')