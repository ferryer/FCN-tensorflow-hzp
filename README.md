<<<<<<< HEAD
# FCN.tensorflow
Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs). 

##主要变化

增加了单张图片测试代码。
added test code of image

代码详解博客：
The implementation is largely based on the reference code provided by the authors of the paper [link](https://blog.csdn.net/qq_40994943/article/details/85042028). 

##使用步骤
Data_produce is used for making your datasets;FCN.py can train your data;Data_zoo are used to hold data,have two folders
,images and annotations .you should divide into training set and test set,the name of original image and the label must
correspond one by one.Please refer to the blog  for specific training steps.[link](https://blog.csdn.net/qq_40994943/article/details/85041493)





