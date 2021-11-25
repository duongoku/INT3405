import json
import math
import pandas as pd
import re
import requests
import unicodedata

ROOT_DATA = 'shopee_sentiment_data_set'


def preview_file():
    global ROOT_DATA
    data = pd.read_csv(f'{ROOT_DATA}/train_preprocess_unsegment.csv')
    print(data['text'].tail)
    print(data['preprocess_text'].tail)
    data = pd.read_csv(f'{ROOT_DATA}/p_train.csv')
    print(data['text'].tail)


def rm_accents_spaces(s: str):
    # the accents looks pretty cool right lol
    accents = ['̀', '̃', '́', '̉', '̣']

    i = 1
    while i < len(s):
        if s[i] in accents:
            s = f'{s[:i-1]}{s[i]}{s[i+2:]}'
        i += 1

    return s


def segment_word(word: str, word_dict: dict):
    result = []
    space = []
    cost = [0]
    for i in range(1, len(word)+1):
        min_cost = float('inf')
        p = 0

        for j in range(1, i+1):
            w = word[max(0, i-j):i]
            w = word_dict.get(w, {'appearance': 0, 'cost': float('inf')})
            c = cost[max(0, i-j)] + w['cost']
            if min_cost > c:
                min_cost = c
                p = max(0, i-j)

        space.append(p)
        cost.append(min_cost)

    p = len(space)-1
    while(p >= 0):
        result.append(word[space[p]:p+1])
        p = space[p]-1

    result.reverse()

    return result


def segment_sentence(sentence: str, word_dict: dict):
    words = sentence.split(' ')
    result = []
    for word in words:
        result.extend(segment_word(word, word_dict))
    result = ' '.join(result)
    return result


def load_word_dict(cached: bool = False):
    # Check if cached
    if cached:
        with open(f'{ROOT_DATA}/word_dict.json', 'r', encoding='utf8') as f:
            word_dict = json.load(f)
    else:
        # Get the online data
        print('Fetching word list . . .')
        url = 'https://raw.githubusercontent.com/garfieldnate/vi_experiments/master/wiki_word_list/wikipedia_unigrams.txt'
        response = requests.get(url)

        # Parse the data
        print('Processing word list . . .')
        raw = response.text
        lines = raw.split('\n')
        lines = lines[1:]
        word_dict = {}
        total_count = 0
        for line in lines:
            tmp = line.split('\t')
            if len(tmp) == 2:
                appearance = int(tmp[1])
                total_count += appearance
                word_dict[re.sub(r'\s+', '_', tmp[0])] = {
                    'appearance': appearance,
                    'cost': 0
                }

        for i in range(101):
            word_dict[str(i)] = {
                'appearance': 100,
                'cost': 0
            }
            total_count += 100

        for word in word_dict:
            word_dict[word]['cost'] = math.log(
                total_count/word_dict[word]['appearance']
            )

        # Cache the data
        print('Caching word list . . .')
        with open(f'{ROOT_DATA}/word_dict.json', 'w+', encoding='utf8') as f:
            json.dump(word_dict, f)

    return word_dict


def preprocess(filename: str = 'train.csv', field: str = 'text', underscore_mode: bool = False):
    global ROOT_DATA
    data = pd.read_csv(f'{ROOT_DATA}/{filename}')

    reviews = data[field]
    p_reviews = []

    count = 0
    word_list = load_word_dict()

    for review in reviews:
        count += 1
        if count % 1000 == 1:
            print(f'Processing row {count}-{min(len(reviews), count+999)}')

        review = review.lower()
        review = re.sub(r'\s+', ' ', review)
        review = rm_accents_spaces(review)
        review = unicodedata.normalize('NFC', review)
        review = re.sub(r'\s*\W\s*', ' ', review)
        review = re.sub(r'\s+', ' ', review)
        review = segment_sentence(review, word_list)
        if underscore_mode is False:
            review = re.sub(r'_', ' ', review)
        p_reviews.append(review)

    data[field] = p_reviews
    data.to_csv(f'{ROOT_DATA}/p_{filename}')


if __name__ == '__main__':
    preprocess('train.csv', 'text')
    preview_file()
