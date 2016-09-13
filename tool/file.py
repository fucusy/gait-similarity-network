__author__ = 'fucus'

import sys
sys.path.append('../')
from config import Project as p
import logging
import time
import os
import numpy as np

def generate_result_file(name, y_result):
    if len(name) != len(y_result):
        print("error the len of name:%d do not equal the len of y_result:%d" % (len(name), len(y_result)))
        exit()

    y_result_matrix =  y_result

    file_name = time.strftime("%Y_%m_%d__%H_%M.csv")
    output_path = p.result_output_path.strip()
    final_path = ""
    if len(output_path) > 0 and output_path[0] == '/':
        final_path = output_path
    else:
        final_path = "%s/%s" % (p.project_path, output_path)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    file_obj = open("%s/%s" %(final_path, file_name), "w")
    file_obj.write(','.join(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']) + '\n')
    for idx in range(len(name)):
        file_obj.write("%s," % os.path.basename(name[idx]))
        num_to_str_result = ["%f" % x for x in y_result_matrix[idx]]
        file_obj.write(','.join(num_to_str_result))
        file_obj.write('\n')
    file_obj.close()
    logging.info("write result to %s" % file_name)

if __name__ == '__main__':
    name = np.array(['test.jpg', '/home/cq/test2.jpg']);
    y_result = np.array([
        [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
        ], dtype=np.float32);
    generate_result_file(name, y_result)
