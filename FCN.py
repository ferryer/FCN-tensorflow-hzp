from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import scipy.misc as misc
import time
batch_size = 2  # batch 大小
logs_dir = "logs/"
data_dir = "Data_zoo/MIT_SceneParsing/"  # 存放数据集的路径，需要提前下载
data_name = "ADEChallengeData2016"
learning_rate = 1e-4  # 学习率
model_path = "Model_zoo/imagenet-vgg-verydeep-19.mat"  # VGG网络参数文件，需要提前下载
debug = False
mode = 'test'  # 训练模式train | visualize

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'  # 训练好的VGGNet参数

MAX_ITERATION = 1000  # 最大迭代次数
NUM_OF_CLASSESS = 2  # 类的个数
IMAGE_SIZE = 224  # 图像尺寸


# 根据载入的权重建立原始的 VGGNet 的网络
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")  # k值转置
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")  # b转成一维
            current = utils.conv2d_basic(current, kernels, bias)  # 图像卷积完加b
            print("当前形状：", np.shape(current))
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current) # 平均池化
            print("当前形状：", np.shape(current))
        net[name] = current
    return net


# FCN的网络结构定义，网络中用到的参数是迁移VGG训练好的参数
def inference(image, keep_prob):  # 输入图像和dropout值
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 加载模型数据，获得标准化均值
    print("原始图像：", np.shape(image))
    model_data = utils.get_model_data(model_path)
    mean = model_data['normalization'][0][0][0]  # 通过字典获取mean值，vgg模型参数里有normaliza这个字典，三个0用来去虚维找到mean
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])  # 从数组的形状中删除单维度条目，获得vgg权重

    # 图像预处理
    processed_image = utils.process_image(image, mean_pixel)  # 图像减平均值实现标准化
    print("预处理后的图像:", np.shape(processed_image))

    with tf.variable_scope("inference"):
        # 建立原始的VGGNet-19网络

        print("开始建立VGG网络：")
        image_net = vgg_net(weights, processed_image)

        # 在VGGNet-19之后添加 一个池化层和三个卷积层
        conv_final_layer = image_net["conv5_3"]  # 14*14*512
        print("VGG处理后的图像：", np.shape(conv_final_layer))

        pool5 = utils.max_pool_2x2(conv_final_layer)  # w，h/32 =7*7*512

        print("pool5：", np.shape(pool5))

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)  # 1*1*4096
        relu6 = tf.nn.relu(conv6, name="relu6")
        if debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        print("conv6:", np.shape(relu_dropout6))

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)  # 1*1*4096
        relu7 = tf.nn.relu(conv7, name="relu7")
        if debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        print("conv7:", np.shape(relu_dropout7))

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)  # 第8层卷积层 分类2类 1*1*2

        print("conv8:", np.shape(conv8))
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # 对卷积后的结果进行反卷积操作
        deconv_shape1 = image_net["pool4"].get_shape()  # 将pool4 即1/16结果尺寸拿出来 做融合 [b,h,w,c]
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")# 扩大两倍  所以stride = 2  kernel_size = 4
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))  #14*14*512
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")  # 将pool4和conv_t1拼接，逐像素相加

        print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))  # (14, 14, 512)

        deconv_shape2 = image_net["pool3"].get_shape()  # 获得pool3尺寸 是原图大小的1/8
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")# 输出通道数为pool3通道数，输入通道数为pool4通道数
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))# 将上一层融合结果fuse_1在扩大两倍，输出尺寸和pool3相同
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))  # (28, 28, 256)

        shape = tf.shape(image)  # 获得原始图像大小
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])  # 矩阵拼接
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        print("conv_t3:", [np.shape(image)[1], np.shape(image)[2], NUM_OF_CLASSESS])  # (224,224,2)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")  # (224,224,1)目前理解是每个像素点所有通道取最大值

    return tf.expand_dims(annotation_pred, dim=3), conv_t3   # 从第三维度扩展形成[b,h,w,c] 其中c=1，即224*224*1*1


# 返回优化器
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


# 主函数
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    print("setting up vgg initialized conv layers ...")

    # 定义好FCN的网络模型
    pred_annotation, logits = inference(image, keep_probability)
    # 定义损失函数，这里使用交叉熵的平均值作为损失函数
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    # 返回需要训练的变量列表
    trainable_var = tf.trainable_variables()

    if debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    # 加载数据集
    print("Setting up image reader...")

    train_records, valid_records = scene_parsing.read_dataset(data_dir, data_name)

    print("训练集的大小:", len(train_records))
    print("验证集的大小:", len(valid_records))

    print("Setting up dataset reader")
    
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}

    if mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)  # 读取图片 产生类对象 其中包含所有图片信息
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    # 开始训练模型
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()  # 保存模型类实例化

    sess.run(tf.global_variables_initializer())  # 变量初始化
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:  # 如果存在checkpoint文件 则恢复sess
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
            print(np.shape(train_images), np.shape(train_annotations))
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)
            print("step:", itr)

            if itr % 10 == 0:
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                saver.save(sess, logs_dir + "model.ckpt", itr)  # 保存模型

    elif mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(batch_size)

        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,  # 预测结果
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), logs_dir, name="inp_" + str(5 + itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), logs_dir, name="gt_" + str(5 + itr))
            utils.save_image(pred[itr].astype(np.uint8), logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)

    else:  # 测试模式
        since = time.time()  # 时间模块

        test_image = misc.imread('G:\\yuantu8.jpg')
        resize_image = misc.imresize(test_image, [224, 224], interp='nearest')
        a = np.expand_dims(resize_image, axis=0)
        a = np.array(a)

        pred = sess.run(pred_annotation, feed_dict={image: a, keep_probability: 1.0})  # 预测测试结果

        pred = np.squeeze(pred, axis=3)  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
        # utils.save_image(pred[0].astype(np.uint8), logs_dir, name="pred_" + str(5))
        utils.save_image(pred[0].astype(np.uint8),'G:/2/', name="pred_" + str(5))
        print("Saved image: succeed")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 打印出来时间



if __name__ == "__main__":
    tf.app.run()
