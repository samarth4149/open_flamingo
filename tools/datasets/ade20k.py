import glob
import os
import cv2
import numpy as np

# read in the annotation file
annotation_folder_path = '/projectnb/ivc-ml/sunxm/datasets/ADEChallengeData2016/annotations/validation'
lines = []
for file in glob.glob(annotation_folder_path+'/*.png'):
    img = cv2.imread(file)
    labels = np.unique(img)
    labels_str = [str(a) for a in labels]
    base_name = os.path.basename(file)
    line = base_name + ',' + ','.join(labels_str) + '\n'
    lines.append(line)

annotation_file_path = '/projectnb/ivc-ml/sunxm/datasets/ADEChallengeData2016/annotations/validation.txt'
with open(annotation_file_path, 'w+') as f:
    f.writelines(lines)