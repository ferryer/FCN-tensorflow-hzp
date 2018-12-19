#执行后的结果会生成对应图片的文件夹，里面包括四个文件：img，info，label，label_viz
# labelme_json_dataset生成的标注图像文件是每个json对应一个文件夹，写了以下代码来进行批量获取label.png文件
#只需将GT_from_PATH设置为所有json文件夹所在根目录即可


import os
import random
import shutil
import re

GT_from_PATH = "F:/Graduate"
GT_to_PATH = "F:/Graduate/"

def copy_file(from_dir, to_dir, Name_list):
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)
    for name in Name_list:
        try:
            # print(name)
            if not os.path.isfile(os.path.join(from_dir, name)):
                print("{} is not existed".format(os.path.join(from_dir, name)))
            shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, name))
            # print("{} has copied to {}".format(os.path.join(from_dir, name), os.path.join(to_dir, name)))
        except:
            # print("failed to move {}".format(from_dir + name))
            pass
        # shutil.copyfile(fileDir+name, tarDir+name)
    print("{} has copied to {}".format(from_dir, to_dir))


if __name__ == '__main__':
    filepath_list = os.listdir(GT_from_PATH)
    # print(name_list)
    for i, file_path in enumerate(filepath_list):
        gt_path = "{}/{}_gt.png".format(os.path.join(GT_from_PATH, filepath_list[i]), file_path[:-5])
        print("copy {} to ...".format(gt_path))
        gt_name = ["{}_gt.png".format(file_path[:-5])]
        gt_file_path = os.path.join(GT_from_PATH, file_path)
        copy_file(gt_file_path, GT_to_PATH, gt_name)
