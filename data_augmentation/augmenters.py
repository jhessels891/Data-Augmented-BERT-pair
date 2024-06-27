import random
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from categories import categories_methods
import shutil
from transformers import MarianMTModel, MarianTokenizer


#Implement KA
def keyboard_augmentation(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k):
    df = categories_methods.create_df(xml_data)
    cat_freq = categories_methods.get_category_frequencies(df)
    low_frequencies = categories_methods.get_low_frequencies(cat_freq, low_freq_min, low_freq_max)
    number_of_extra_sentences = k - 1

    for index, row in df.iterrows():
        sentence_id = row['Sentence_ID']
        sentence_text = row['Sentence_Text']
        category = row['Category']
        polarity = row['Polarity']

        if category in low_frequencies:
            for i in range(0, number_of_extra_sentences):
                aug_char_p = random.uniform(0.1, 0.2)
                aug_word_p = random.uniform(0.2, 0.3)

                aug = nac.KeyboardAug(aug_char_p=aug_char_p, aug_word_p=aug_word_p, include_upper_case=False,
                                      include_special_char=False)

                new_sentence_id = sentence_id + "ka" + str(i + 1)
                new_sentence_text = aug.augment(sentence_text)

                data = f"{new_sentence_id}\t{polarity}\t{category}\t{new_sentence_text[0]}\n"
                add_data_to_csv(augmented_csv_file, data)


#Implement BT
def backtranslation(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k):
    df = categories_methods.create_df(xml_data)
    cat_freq = categories_methods.get_category_frequencies(df)
    low_frequencies = categories_methods.get_low_frequencies(cat_freq, low_freq_min, low_freq_max)

    # Change 'fr' to 'zh' to switch to Chinese backtranslation
    model_name_en_to_xx = "Helsinki-NLP/opus-mt-en-fr"
    model_name_xx_to_en = "Helsinki-NLP/opus-mt-fr-en"

    for index, row in df.iterrows():
        sentence_id = row['Sentence_ID']
        sentence_text = row['Sentence_Text']
        category = row['Category']
        polarity = row['Polarity']

        if category in low_frequencies:

            for i in range(0, k - 1):
                new_sentence_id = sentence_id + "bt" + str(i + 1)
                translated_sentence = translate_text(sentence_text, model_name_en_to_xx)
                sentence_text = translate_text(translated_sentence, model_name_xx_to_en)
                data = f"{new_sentence_id}\t{polarity}\t{category}\t{sentence_text}\n"
                add_data_to_csv(augmented_csv_file, data)


#Translate text using MarianMT model
def translate_text(text, model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_text[0]


#Implement EDA
def easy_data_augmentation(xml_data, augmented_csv_file, low_freq_min, low_freq_max, k):
    df = categories_methods.create_df(xml_data)
    cat_freq = categories_methods.get_category_frequencies(df)
    low_frequencies = categories_methods.get_low_frequencies(cat_freq, low_freq_min, low_freq_max)
    number_of_extra_sentences = k - 1

    for index, row in df.iterrows():
        sentence_id = row['Sentence_ID']
        sentence_text = row['Sentence_Text']
        category = row['Category']
        polarity = row['Polarity']

        if category in low_frequencies:
            for i in range(0, number_of_extra_sentences):
                alpha = random.randint(1, 4)
                aug_p = random.uniform(0.2, 0.3)

                new_sentence_id = sentence_id + "eda" + str(i + 1)
                new_sentence_text = []

                # 1 equals Synonym Replacement: synonym aug
                if alpha == 1:
                    aug = naw.SynonymAug(aug_p=aug_p)
                    new_sentence_text = aug.augment(sentence_text)

                # 2 equals Random Insertion: context word emb aug
                elif alpha == 2:
                    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert', aug_p=aug_p)
                    new_sentence_text = aug.augment(sentence_text)

                # 3 equals Random Swap
                elif alpha == 3:
                    aug = naw.RandomWordAug(action='swap', aug_p=aug_p)
                    new_sentence_text = aug.augment(sentence_text)

                # 4 equals Random Deletion
                elif alpha == 4:
                    aug = naw.RandomWordAug(action='delete', aug_p=aug_p)
                    new_sentence_text = aug.augment(sentence_text)

                data = f"{new_sentence_id}\t{polarity}\t{category}\t{new_sentence_text[0]}\n"
                add_data_to_csv(augmented_csv_file, data)


#Add the newly created pairs to the data file
def add_data_to_csv(csv_file, data):
    with open(csv_file, 'a', newline='') as csvfile:
        csvfile.write("".join(data))


def copy_csv_file(csv_file, augmented_csv_file):
    shutil.copyfile(csv_file, augmented_csv_file)
