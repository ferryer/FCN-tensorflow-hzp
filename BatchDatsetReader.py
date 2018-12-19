# coding=utf-8
import numpy as np
import scipy.misc as misc


# 批量读取数据集的类
class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
          Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
          sample record:
           {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
          Available options:
            resize = True/ False
            resize_size = #size of output image - does bilinear resize
            color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True

        # 读取训练集图像
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False

        # 读取label的图像，由于label图像是二维的，这里需要扩展为三维
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print(self.images.shape)
        print(self.annotations.shape)

    # 把图像转为 numpy数组
    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')  # 使用最近邻插值法resize图片
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations  # 返回图片和标签全路径

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset  # 当前第几个batch
        self.batch_offset += batch_size  # 读取下一个batch  所有offset偏移量+batch_size
        if self.batch_offset > self.images.shape[0]:  # 如果下一个batch的偏移量超过了图片总数说明完成了一个epoch
            # Finished epoch
            self.epochs_completed += 1  # epochs完成总数+1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])  # arange生成数组(0 - len-1) 获取图片索引
            np.random.shuffle(perm)  # 对图片索引洗牌
            self.images = self.images[perm]  # 洗牌之后的图片顺序
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0  # 下一个epoch从0开始
            self.batch_offset = batch_size  # 已完成的batch偏移量

        end = self.batch_offset   # 开始到结束self.batch_offset   self.batch_offset+batch_size
        return self.images[start:end], self.annotations[start:end]  # 取出batch

    def get_random_batch(self, batch_size):  # 按照一个batch_size一个块，进行对所有图片总数进行随机操作，相当于洗牌工作
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]


