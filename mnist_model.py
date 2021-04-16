
from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #载入数据
# 搭建LeNet网络
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network
network = LeNet()
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
network.summary() #使用summary函数输出param

''' 
reshape函数可以把二维的矩阵压缩成一维的向量，
后面把数据类型转换为浮点型是因为接下来的运算会涉及梯度和权重，这些都是小数值
'''
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
'''
标签，这是一个十分类问题，最后一层的神经元有10个
此处需要进行one-hot编码
独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码
，每个状态都有它独立的寄存器位
'''
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
start1 = time.clock()
network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
print(time.clock() - start1)
start2 = time.clock()
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
print(time.clock() - start2)
network.save('mnist_lnn.h5')

'''
# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, test_images.shape)
print(train_images[0])
print(train_labels[0])
plt.imshow(train_images[0])
plt.show()
'''