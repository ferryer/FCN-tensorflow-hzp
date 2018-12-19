#批量重命名图片和json格式的样本
import os
path = "D:\\tu"
filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
i=1
for file in filelist: #遍历所有文件
    if file.endswith('.jpg'):
      Olddir=os.path.join(path,file) #原来的文件路径
      if os.path.isdir(Olddir):#如果是文件夹则跳过
          continue
      filename=os.path.splitext(file)[0] #文件名
      filetype=os.path.splitext(file)[1] #文件扩展名
      Newdir=os.path.join(path,str(i).zfill(6)+filetype) #用字符串函数zfill 以0补全所需位数
      os.rename(Olddir,Newdir)#重命名
    if file.endswith('.json'):
        Olddir=os.path.join(path,file) #原来的文件路径
        if os.path.isdir(Olddir): #如果是文件夹则跳过
            continue
        filename=os.path.splitext(file)[0] #文件名
        filetype=os.path.splitext(file)[1] #文件扩展名
        Newdir=os.path.join(path,str(i).zfill(6)+filetype) #用字符串函数zfill 以0补全所需位数
        os.rename(Olddir,Newdir)#重命名
    i = i + 1
