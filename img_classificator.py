import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop

np.random.seed(1337)  # for reproducibility


# CNN模型
def CNNCls(X_train, Y_train, X_test, Y_test):
    # 数据预处理（size：100×100，RGB格式）
    X_train = X_train.reshape(-1, 3, 100, 100) / 255.
    X_test = X_test.reshape(-1, 3, 100, 100) / 255.
    Y_train = np_utils.to_categorical(Y_train, num_classes=2)
    Y_test = np_utils.to_categorical(Y_test, num_classes=2)

    # 建立模型
    model = Sequential()

    # Conv layer 1 output shape (32, 100, 100)
    model.add(Convolution2D(
        batch_input_shape=(None, 3, 100, 100),
        filters=32,  # 滤波器，过滤生成32张图片
        kernel_size=5,  # 2D卷积窗口的宽度和高度，可以是数组
        strides=1,  # 卷积沿宽度和高度方向的步长
        padding='same',  # Padding method 避免数据丢失的东西
        data_format='channels_first',  # 表示输入中维度的顺序(batch, channels, height, width)
    ))
    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (32, 50, 50)
    model.add(MaxPooling2D(
        pool_size=2,  # 沿（垂直，水平）方向缩小比例的因数，=2为缩小一半
        strides=2,  # 表示步长值
        padding='same',  # Padding method
        data_format='channels_first',
    ))

    # Conv layer 2 output shape (64, 50, 50)
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))

    # Pooling layer 2 (max pooling) output shape (64, 25, 25)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # Fully connected layer 1 input shape (64 * 25 * 25) = (40000), output shape (2048)
    model.add(Flatten())  # 把三维flatten成一维
    model.add(Dense(2048))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(2))
    model.add(Activation('softmax'))  # 在多分类的场景中使用广泛。把输入映射为0-1之间的实数，并且归一化保证和为1。

    # Another way to define your optimizer
    adam = Adam(lr=3e-5)  # 一种自适应的梯度下降优化器 https://www.jianshu.com/p/aebcaf8af76e

    # We add metrics to get more results you want to see
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training ------------')
    # Another way to train the model
    model.fit(X_train, Y_train, epochs=3, batch_size=64, )

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, Y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)


# NN模型
def NNCls(X_train, Y_train, X_test, Y_test):
    # 数据预处理（size：100×100，RGB格式）
    X_train = X_train / 255.
    X_test = X_test / 255.
    Y_train = np_utils.to_categorical(Y_train, num_classes=2)
    Y_test = np_utils.to_categorical(Y_test, num_classes=2)

    # 建立模型
    model = Sequential([
        Dense(32, input_dim=30000),
        Activation('relu'),  # 卷积层推荐relu，循环推荐relu或tanh，还有sigma之类的
        Dense(2),
        Activation('softmax'),
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)  # Ir指学习率
    # https://blog.csdn.net/willduan1/article/details/78070086

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',  # 交叉熵是用来评估当前训练得到的概率分布与真实分布的差异情况。
                  # 它刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。
                  # 相较于MSE，梯度下降速度会更快
                  metrics=['accuracy'])

    print('Training ------------')
    # Another way to train the model
    model.fit(X_train, Y_train, epochs=3, batch_size=16)  # 每次训练32个样本，整体共训练2次

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, Y_test)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)


# 加载数据
X = np.load("imgDataMatrix.npy", allow_pickle=True)
Y = np.load("imgNameMatrix.npy", allow_pickle=True)
# 随机分类train和test（8:2）
ran = np.random.random(size=(len(Y), 1))
trainList = ran > 0.2
testList = ran <= 0.2
dataRow = np.concatenate((X, Y, trainList), axis=1)
X_train = dataRow[dataRow[:, X.shape[1]+1] == True][:, 0:X.shape[1]]
Y_train = dataRow[dataRow[:, X.shape[1]+1] == True][:, X.shape[1]]
X_test = dataRow[dataRow[:, X.shape[1]+1] == False][:, 0:X.shape[1]]
Y_test = dataRow[dataRow[:, X.shape[1]+1] == False][:, X.shape[1]]

# 选择模型
NNCls(X_train, Y_train, X_test, Y_test)       # 算的更快，但准确率更低
# CNNCls(X_train, Y_train, X_test, Y_test)
