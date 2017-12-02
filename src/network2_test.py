# -*- coding: utf-8 -*-
# @Time    : 2017/12/2 下午6:45
# @Author  : KaWa
# @File    : network2_test.py
# @Project : neural-networks-and-deep-learning
# @Copyright(c) 2017 By KaWa All rights reserved.

# library
# standard library

# third-party library
import network2
import mnist_loader

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10])

    # L2 & L1
    # net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, mode="L2", evaluation_data=validation_data, monitor_evaluation_accuracy=True)
    # net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, mode="L1", evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    # early stop
    net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, mode="L2", evaluation_data=validation_data, max_try=1, monitor_evaluation_accuracy=True)