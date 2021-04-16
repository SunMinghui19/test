from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from keras.datasets import mnist
import time
my_model = load_model('F:/2020上半年工作资料/K8S学习/论文/异构AI/本机项目/mnist_lnn.h5')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

start = time.clock()
start1 = time.time()
test_loss, test_accuracy = my_model.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
print(time.clock()-start)
print(time.time()-start1)

'''
pred = my_model.predict(test_images[:])
print('Label of testing sample', np.argmax(test_labels))
print('Output of the softmax layer', pred[0])
print('Network prediction:', np.argmax([pred[0]]))
'''