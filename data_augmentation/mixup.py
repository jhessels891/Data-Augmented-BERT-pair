import random
import numpy as np
import pandas as pd
import shutil
from categories import categories_methods

""" Mixup without embeddings """
def text_mixup(sentence1, sentence2, mixup_beta):
    # Calculate the mixup factor lambda from Beta distribution
    labda = np.random.beta(mixup_beta, mixup_beta)

    # Split sentences into words
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Determine the length of the mixed sentence
    len_sentence = max(len(words1), len(words2))

    # Create the new mixed sentence
    new_sentence = []
    for i in range(len_sentence):
        if i < len(words1) and i < len(words2):
            if random.random() < labda:
                new_sentence.append(words1[i])
            else:
                new_sentence.append(words2[i])
        elif i < len(words1):
            new_sentence.append(words1[i])
        else:
            new_sentence.append(words2[i])

    return " ".join(new_sentence)


# Add the newly created pairs to the data file
def add_data_to_csv(csv_file, data):
    with open(csv_file, 'a', newline='') as csvfile:
        csvfile.write(data)


# Copy original CSV file to augmented file
def copy_csv_file(csv_file, augmented_csv_file):
    shutil.copyfile(csv_file, augmented_csv_file)


# Implement mixup augmentation
def mixup_augmentation(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k, mixup_beta):
    df = categories_methods.create_df(xml_data)
    cat_freq = categories_methods.get_category_frequencies(df)
    low_frequencies = categories_methods.get_low_frequencies(cat_freq, low_freq_min, low_freq_max)

    # Prepare sentences for mixup
    low_freq_sentences = df[df['Category'].isin(low_frequencies)]

    for index, row in low_freq_sentences.iterrows():
        sentence_id = row['Sentence_ID']
        sentence_text = row['Sentence_Text']
        category = row['Category']
        polarity = row['Polarity']

        for i in range(k - 1):
            # Select another random low-frequency sentence with the same category
            rand_row = low_freq_sentences[low_freq_sentences['Category'] == category].sample(1).iloc[0]
            rand_sentence_text = rand_row['Sentence_Text']

            # Print the original sentences being mixed
            print(i)
            print(f"Original Sentence 1: {sentence_text}")
            print(f"Original Sentence 2: {rand_sentence_text}")

            new_sentence_text = text_mixup(sentence_text, rand_sentence_text, mixup_beta)

            print(new_sentence_text)
            new_sentence_id = f"{sentence_id}_mixup_{i + 1}"
            data = f"{new_sentence_id}\t{polarity}\t{category}\t{new_sentence_text}\n"
            add_data_to_csv(augmented_csv_file, data)


# Main augmentation function to include mixup
def mixup(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k):

    # Perform mixup augmentation
    mixup_beta = 0.2
    mixup_augmentation(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k, mixup_beta)
