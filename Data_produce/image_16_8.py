# import cv2
# from PIL import Image
# import numpy as np
# img = Image.open(open('E:/dev/AI/jupyter/Mask_RCNN-master/labelme-master\input/1_json/label_old.png'))
# img = np.asarray(img)
# img = Image.fromarray(np.uint8(img))
# img.save('E:/dev/AI/jupyter/Mask_RCNN-master/labelme-master\input/1_json/label.png')


from PIL import Image
import numpy as np
import math
import os

path = 'F:/Graduate/'
newpath = 'F:/Graduate/'


def toeight():
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
        img = np.array(img)
        # img = Image.fromarray(np.uint8(img / float(math.pow(2, 16) - 1) * 255))
        img = Image.fromarray(np.uint8(img))
        img.save(newpath + file)


toeight()