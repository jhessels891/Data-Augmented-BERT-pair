import os
import re


def generate_NLI(data_xml, file_name):
    data_dir = '../data/semevaldata/'

    dir_path = data_dir + 'bert-pair/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(dir_path + file_name, "w", encoding="utf-8") as g:
        with open(
                data_dir + data_xml,
                "r", encoding="utf-8") as f:
            s = f.readline().strip()
            while s:
                category = []
                polarity = []
                if "<sentence id" in s:
                    left = s.find("id")
                    right = s.find(">")
                    id = s[left + 4:right - 1]
                    while not "</sentence>" in s:
                        if "<text>" in s:
                            left = s.find("<text>")
                            right = s.find("</text>")
                            text = s[left + 6:right]
                        if "Opinion" in s:
                            left = s.find("category=")
                            right = s.find("polarity=")
                            category.append(s[left + 10:right - 2])
                            left = s.find("polarity=")
                            right = s.find(" from")
                            polarity.append(s[left + 10:right - 2])
                        s = f.readline().strip()
                    if "FOOD#QUALITY" in category:
                        g.write(id + "\t" + polarity[
                            category.index("FOOD#QUALITY")] + "\t" + "FOOD#QUALITY" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "FOOD#QUALITY" + "\t" + text + "\n")
                    if "SERVICE#GENERAL" in category:
                        g.write(id + "\t" + polarity[category.index(
                            "SERVICE#GENERAL")] + "\t" + "SERVICE#GENERAL" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "SERVICE#GENERAL" + "\t" + text + "\n")
                    if "RESTAURANT#GENERAL" in category:
                        g.write(id + "\t" + polarity[category.index(
                            "RESTAURANT#GENERAL")] + "\t" + "RESTAURANT#GENERAL" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "RESTAURANT#GENERAL" + "\t" + text + "\n")
                    if "AMBIENCE#GENERAL" in category:
                        g.write(id + "\t" + polarity[
                            category.index("AMBIENCE#GENERAL")] + "\t" + "AMBIENCE#GENERAL" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "AMBIENCE#GENERAL" + "\t" + text + "\n")
                    if "FOOD#STYLE_OPTIONS" in category:
                        g.write(id + "\t" + polarity[
                            category.index(
                                "FOOD#STYLE_OPTIONS")] + "\t" + "FOOD#STYLE_OPTIONS" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "FOOD#STYLE_OPTIONS" + "\t" + text + "\n")
                    if "RESTAURANT#MISCELLANEOUS" in category:
                        g.write(id + "\t" + polarity[
                            category.index("RESTAURANT#MISCELLANEOUS")] + "\t" + "RESTAURANT#MISCELLANEOUS" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "RESTAURANT#MISCELLANEOUS" + "\t" + text + "\n")
                    if "FOOD#PRICES" in category:
                        g.write(id + "\t" + polarity[
                            category.index("FOOD#PRICES")] + "\t" + "FOOD#PRICES" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "FOOD#PRICES" + "\t" + text + "\n")
                    if "RESTAURANT#PRICES" in category:
                        g.write(id + "\t" + polarity[
                            category.index("RESTAURANT#PRICES")] + "\t" + "RESTAURANT#PRICES" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "RESTAURANT#PRICES" + "\t" + text + "\n")
                    if "DRINKS#QUALITY" in category:
                        g.write(id + "\t" + polarity[
                            category.index("DRINKS#QUALITY")] + "\t" + "DRINKS#QUALITY" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "DRINKS#QUALITY" + "\t" + text + "\n")
                    if "DRINKS#STYLE_OPTIONS" in category:
                        g.write(id + "\t" + polarity[category.index(
                            "DRINKS#STYLE_OPTIONS")] + "\t" + "DRINKS#STYLE_OPTIONS" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "DRINKS#STYLE_OPTIONS" + "\t" + text + "\n")
                    if "LOCATION#GENERAL" in category:
                        g.write(id + "\t" + polarity[
                            category.index("LOCATION#GENERAL")] + "\t" + "LOCATION#GENERAL" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "LOCATION#GENERAL" + "\t" + text + "\n")
                    if "DRINKS#PRICES" in category:
                        g.write(id + "\t" + polarity[
                            category.index("DRINKS#PRICES")] + "\t" + "DRINKS#PRICES" + "\t" + text + "\n")
                    else:
                        g.write(id + "\t" + "none" + "\t" + "DRINKS#PRICES" + "\t" + text + "\n")

                else:
                    s = f.readline().strip()


