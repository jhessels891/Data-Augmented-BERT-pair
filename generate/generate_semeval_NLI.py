import csv

from generator_method import generate_NLI
import xml.etree.ElementTree as ET

# Generate the BERT-pair data input
generate_NLI("ABSA-16_SB1_Restaurants_Test_Gold.xml", "pre_test_NLI_16.csv")
generate_NLI("ABSA-16_SB1_Restaurants_Train_Data.xml", "pre_train_NLI_16.csv")

input_file = '../data/semevaldata/bert-pair/pre_test_NLI_16.csv'
output_file = '../data/semevaldata/bert-pair/test_NLI_16.csv'


with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in reader:
        modified_row = [cell.replace('positiv', 'positive').replace('negativ', 'negative').replace('neutra', 'neutral') for cell in row]
        writer.writerow(modified_row)

input_file = '../data/semevaldata/bert-pair/pre_train_NLI_16.csv'
output_file = '../data/semevaldata/bert-pair/train_NLI_16.csv'


with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in reader:
        modified_row = [cell.replace('positiv', 'positive').replace('negativ', 'negative').replace('neutra', 'neutral') for cell in row]
        writer.writerow(modified_row)


