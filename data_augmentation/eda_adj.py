import csv
import math
import random
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from categories import categories_methods
import shutil
# for the first time you use wordnet
# nltk.download('wordnet')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from pywsd import simple_lesk


#Implement adjusted EDA
def easy_data_augmentation_adjusted(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k):
    df = categories_methods.create_df(xml_data)
    cat_freq = categories_methods.get_category_frequencies(df)
    low_frequencies = categories_methods.get_low_frequencies(cat_freq, low_freq_min, low_freq_max)

    for index, row in df.iterrows():
        sentence_id = row['Sentence_ID']
        sentence_text = row['Sentence_Text']
        category = row['Category']
        polarity = row['Polarity']

        if category in low_frequencies:
            for i in range(0, k - 1):
                alpha = random.randint(1, 4)
                aug_p = random.uniform(0.2, 0.3)

                new_sentence_id = sentence_id + "eda_adj" + str(i + 1)
                words = sentence_text.split(' ')

                # 1 equals Synonym Replacement
                if alpha == 1:
                    new_sentence_text = ' '.join(synonym_replacement_adj(words, aug_p))

                # 2 equals Random Insertion 
                elif alpha == 2:
                    new_sentence_text = ' '.join(random_insertion_adj(words, aug_p))

                # 3 equals Random Swap
                elif alpha == 3:
                    new_sentence = []
                    aug = naw.RandomWordAug(action='swap', aug_p=aug_p)
                    new_sentence = aug.augment(sentence_text)
                    new_sentence_text = new_sentence[0]

                # 4 equals Random Deletion
                elif alpha == 4:
                    # new_sentence = []
                    aug = naw.RandomWordAug(action='delete', aug_p=aug_p)
                    new_sentence = aug.augment(sentence_text)
                    new_sentence_text = new_sentence[0]

                data = f"{new_sentence_id}\t{polarity}\t{category}\t{new_sentence_text}\n"
                add_data_to_csv(augmented_csv_file, data)


# Synonym replacement
def synonym_replacement_adj(words, aug_p):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words and word != '$t$']))
    random.shuffle(random_word_list)
    n = int(aug_p * len(random_word_list))
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms_adjusted(words, random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
            print("replaced", random_word, "with", synonym)
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms_adjusted(words, random_word):
    pos_tags = nltk.pos_tag(words)
    for word, func in pos_tags:
        if word == random_word:
            if not get_wordnet_pos(func):
                return []
            meaning = simple_lesk(' '.join(words), random_word, pos=get_wordnet_pos(func))
    synonyms = []
    if meaning:
        for syn in meaning.lemma_names():
            synonym = syn.lower()
            synonyms.append(synonym)
        if random_word in synonyms:
            synonyms.remove(random_word)
    return synonyms


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

    # Random insertion


def random_insertion_adj(words, aug_p):
    new_words = words.copy()
    for _ in range(int(aug_p * len(words))):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms_adjusted(new_words, random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


#Add the newly created pairs to the data file
def add_data_to_csv(csv_file, data):
    with open(csv_file, 'a', newline='') as csvfile:
        csvfile.write("".join(data))


# stop words list
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ''}
