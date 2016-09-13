#!/usr/bin/python2.7

import os
import shutil

source_path = "/Volumes/Passport/data/CASIA_full_gait_data_set/DatasetB/silhouettes"
target_path = "/Volumes/Passport/data/gait-simple-cnn-data"

gait_type = "nm"
gait_seq = ["01", "02", "03", "04", "05", "06"]
angles = ["090"]

for person_id in range(1, 125):
    id_target_dir = "%s/%03d" % (target_path, person_id)

    if not os.path.exists(id_target_dir):
        os.makedirs(id_target_dir)

    for seq in gait_seq:
        for angle in angles:
            id_source_dir = "%s/%03d/%s-%s/%s/" \
                % (source_path, person_id, gait_type, seq, angle)
            print("copying data to %s from %s" % (id_target_dir, id_source_dir))
            for img in os.listdir(id_source_dir):
                img_path = "%s/%s" % (id_source_dir, img)
                shutil.copy(img_path, id_target_dir)
