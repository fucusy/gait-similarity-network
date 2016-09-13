__author__ = 'fucus'

import logging
from config import Project
import sys
from shutil import copy2
import os
from shutil import rmtree
import numpy as np
from tool.file import generate_result_file

def restructure_img(result_csv_file_path, output_img_path):
    logging.info("restructure the image file by %s file to %s" % (result_csv_file_path, output_img_path))
    for j in range(10):
        output_img_type_path = "%s/c%d" % (output_img_path, j)
        if not os.path.exists(output_img_type_path):
            os.makedirs(output_img_type_path)

    count = 0
    for line in open(result_csv_file_path):
        if count % 1000 == 0:
            logging.info("process %d line of result csv file path now" % count)
        line = line.rstrip("\n")
        if count > 0:
            split_line = line.split(",")
            if len(split_line) != 11:
                logging.warning("can't extract info from line %s:%s" % (count, line))
            else:
                img_path = "%s/%s" % (Project.test_img_folder_path, split_line[0])

                max_score = float(split_line[1])
                max_index = 0
                for j in range(10):
                    if float(split_line[j+1]) > max_score:
                        max_score = float(split_line[j+1])
                        max_index = j
                output_img_type_path = "%s/c%d/" % (output_img_path, max_index)
                copy2(img_path, output_img_type_path)

        count += 1


def average_result(score_dic_list):
    if len(score_dic_list) == 0:
        logging.error("empty score_dic_list")
        return {}

    final_score_dic = {}
    for k in score_dic_list[0].keys():
        tmp_score = None
        for score_dic in score_dic_list:
            if k in score_dic.keys():
                if tmp_score is None:
                    tmp_score = score_dic[k]
                else:
                    tmp_score = np.vstack((tmp_score, score_dic[k]))
            else:
                logging.warning("lack key:%s in score_dic_list" % k)
        final_score_dic[k] = np.mean(tmp_score, axis=0)
    return final_score_dic


def dic_to_file(score_dic):
    file_name_list = []
    result_list = []

    for k in score_dic.keys():
        file_name_list.append(k)
        result_list.append(score_dic[k])
    generate_result_file(file_name_list, result_list)



def file_to_dic(file_name):
    score_dic = {}
    count = 0
    for line in open(file_name):
        count += 1

        # skip first line
        if count == 1:
            continue

        split_line = line.split(",")
        key = split_line[0]
        score = [float(x) for x in split_line[1:]]
        score_dic[key] = np.array(score)
    return score_dic


def merge_files(file_name_list):
    dic_list = []
    for file_name in file_name_list:
        dic_list.append(file_to_dic(file_name))
    final_result = average_result(dic_list)
    dic_to_file(final_result)

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    csv_file_path = "%s/result/67.csv" % Project.project_path
    output_img_path = "%s/../output_img/" % Project.project_path
    #rmtree(output_img_path)
    #restructure_img(csv_file_path, output_img_path)

    result_path = "%s/result/" % Project.project_path
    base_filename_list = ["2016_06_02__20_28-2.csv", "inference_dense_prediction.csv"]
    filename_list = ["%s/%s" % (result_path, x) for x in base_filename_list]

    merge_files(filename_list)