import argparse
import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_y_true(task_name):
    """ 
    Read file to obtain y_true.
    All of five tasks of Sentihood use the test set of task-BERT-pair-NLI-M to get true labels.
    All of five tasks of SemEval-2014 use the test set of task-BERT-pair-NLI-M to get true labels.
    """

    true_data_file = "data/semevaldata/bert-pair/test_NLI_16.csv"

    df = pd.read_csv(true_data_file, sep='\t', header=None).values
    y_true = []
    for i in range(len(df)):
        label = df[i][1]
        assert label in ['positive', 'neutral', 'negative', 'conflict', 'none'], "error!"
        if label == 'positive':
            n = 0
        elif label == 'neutral':
            n = 1
        elif label == 'negative':
            n = 2
        elif label == 'conflict':
            n = 3
        elif label == 'none':
            n = 4
        y_true.append(n)

    return y_true


def get_y_true_lf(task_name):
    y_true_lf = [
        item for index, item in enumerate(get_y_true(task_name)) if index % 12 in [8, 9, 10, 11]
    ]
    return y_true_lf


# for low frequency, we want to have true_data_file only contain the low frequencies, so the last 4 categories that
# repeat every 12 lines

def get_y_pred(task_name, pred_data_dir):
    """
    Read file to obtain y_pred and scores.
    """
    pred = []
    score = []

    if task_name in ["semeval_NLI_M", "semeval_QA_M"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])])
                s = f.readline().strip().split()
    else:
        count = 0
        with open(pred_data_dir + "price.txt", "r", encoding="utf-8") as f_price, \
                open(pred_data_dir + "anecdotes.txt", "r", encoding="utf-8") as f_anecdotes, \
                open(pred_data_dir + "food.txt", "r", encoding="utf-8") as f_food, \
                open(pred_data_dir + "ambience.txt", "r", encoding="utf-8") as f_ambience, \
                open(pred_data_dir + "service.txt", "r", encoding="utf-8") as f_service:
            s = f_price.readline().strip().split()
            while s:
                count += 1
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])])
                if count % 5 == 0:
                    s = f_price.readline().strip().split()
                if count % 5 == 1:
                    s = f_anecdotes.readline().strip().split()
                if count % 5 == 2:
                    s = f_food.readline().strip().split()
                if count % 5 == 3:
                    s = f_ambience.readline().strip().split()
                if count % 5 == 4:
                    s = f_service.readline().strip().split()

    return pred, score


def get_y_pred_lf(task_name, pred_data_dir):
    # Get the predictions and scores using the original function
    pred, score = get_y_pred(task_name, pred_data_dir)
    pred_lf = [
        item for index, item in enumerate(pred) if index % 12 in [8, 9, 10, 11]
    ]
    score_lf = [
        item for index, item in enumerate(score) if index % 12 in [8, 9, 10, 11]
    ]

    return pred_lf, score_lf


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2016.
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0: continue
        s_g = s.intersection(g)
        s_all += len(s)
        g_all += len(g)
        s_g_all += len(s_g)

    p = s_g_all / s_all
    r = s_g_all / g_all
    f = 2 * p * r / (p + r)

    return p, r, f


def semeval_macro_PRF(y_true, y_pred):
    """
    Calculate "Macro P R F" of aspect detection task of SemEval-2014.
    """
    precisions = []
    recalls = []

    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0: continue
        s_g = s.intersection(g)

        s_all = len(s)
        g_all = len(g)
        s_g_all = len(s_g)

        p = s_g_all / s_all if s_all != 0 else 0
        r = s_g_all / g_all if g_all != 0 else 0

        precisions.append(p)
        recalls.append(r)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    macrof = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0

    return avg_precision, avg_recall, macrof


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2016.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] == 4: continue
            total += 1
            tmp = y_pred[i]
            if tmp == 4:
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2] and score[i][0] >= score[i][3]:
                    tmp = 0
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2] and score[i][1] >= score[i][3]:
                    tmp = 1
                elif score[i][2] >= score[i][0] and score[i][2] >= score[i][1] and score[i][2] >= score[i][3]:
                    tmp = 2
                else:
                    tmp = 3
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    elif classes == 3:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3:
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2]:
                    tmp = 0
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2]:
                    tmp = 1
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    else:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3 or y_true[i] == 1: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3 or tmp == 1:
                if score[i][0] >= score[i][2]:
                    tmp = 0
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total

    return sentiment_Acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()

    result = collections.OrderedDict()

    y_true = get_y_true(args.task_name)
    y_true_lf = get_y_true_lf(args.task_name)

    y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
    y_pred_lf, score_lf = get_y_pred_lf(args.task_name, args.pred_data_dir)

    aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
    aspect_P_lf, aspect_R_lf, aspect_F_lf = semeval_PRF(y_true_lf, y_pred_lf)
    aspect_MacroP, aspect_MacroR, aspect_MacroF = semeval_macro_PRF(y_true, y_pred)
    aspect_MacroP_lf, aspect_MacroR_lf, aspect_MacroF_lf = semeval_macro_PRF(y_true_lf, y_pred_lf)
    sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
    sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
    sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
    result = {
        'aspect_MacroP': aspect_MacroP,
        'aspect_MacroR': aspect_MacroR,
        'aspect_MacroF': aspect_MacroF,
        'aspect_MacroP_lf': aspect_MacroP_lf,
        'aspect_MacroR_lf': aspect_MacroR_lf,
        'aspect_MacroF_lf': aspect_MacroF_lf,
    'aspect_P': aspect_P,
    'aspect_R': aspect_R,
    'aspect_F': aspect_F,
    'aspect_P_lf': aspect_P_lf,
    'aspect_R_lf': aspect_R_lf,
    'aspect_F_lf': aspect_F_lf}
    # 'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
    # 'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
    # 'sentiment_Acc_2_classes': sentiment_Acc_2_classes

    for key in result.keys():
        print(key, "=", str(result[key]))


if __name__ == "__main__":
    main()
