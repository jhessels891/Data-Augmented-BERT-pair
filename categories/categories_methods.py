import os
import xml.etree.ElementTree as ET
import openpyxl
import pandas as pd

#Create dataframe from XML file
def create_df(data_xml):
    tree = ET.parse(data_xml)
    root = tree.getroot()

    data = []

    for review in root.findall('Review'):
        rid = review.get('rid')

        for sentence in review.find('sentences').findall('sentence'):
            sentence_id = sentence.get('id')
            sentence_text = sentence.find('text').text if sentence.find('text') is not None else "No text"

            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    data.append([rid, sentence_id, sentence_text, category, polarity])

    df = pd.DataFrame(data, columns=['Review_ID', 'Sentence_ID', 'Sentence_Text', 'Category', 'Polarity'])

    return df

#Turn dataframe back into XML file
def create_xml(df):
    root = ET.Element("Reviews")

    # Group DataFrame rows by Review_ID
    grouped = df.groupby(['Review_ID'])

    for rid, group in grouped:
        review_element = ET.SubElement(root, "Review", rid=str(rid))
        sentences_element = ET.SubElement(review_element, "sentences")

        for _, row in group.iterrows():
            sentence_element = ET.SubElement(sentences_element, "sentence", id=str(row['Sentence_ID']))
            text_element = ET.SubElement(sentence_element, "text")
            text_element.text = row['Sentence_Text']

            if not pd.isnull(row['Category']) and not pd.isnull(row['Polarity']):
                opinions_element = ET.SubElement(sentence_element, "Opinions")
                opinion_element = ET.SubElement(opinions_element, "Opinion", category=row['Category'], polarity=row['Polarity'])

    return ET.tostring(root, encoding='unicode')

#Get the frequencies for aspects in data set
def get_category_frequencies(df):
    category_frequencies = df['Category'].value_counts()
    category_freq_df = pd.DataFrame(category_frequencies.reset_index())
    category_freq_df.columns = ['Category', 'Frequency']

    return category_freq_df

#Get the low frequency aspects in data set
def get_low_frequencies(category_freq_df, range_min, range_max):
    low_freq_categories = category_freq_df[
        (category_freq_df['Frequency'] >= range_min) & (category_freq_df['Frequency'] <= range_max)]

    return low_freq_categories['Category'].tolist()

