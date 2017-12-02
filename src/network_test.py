# -*- coding: utf-8 -*-
# @Time    : 2017/12/2 下午5:35
# @Author  : KaWa
# @File    : network_test.py
# @Project : neural-networks-and-deep-learning
# @Copyright(c) 2017 By KaWa All rights reserved.

# library
# standard library

# third-party library
import network
import mnist_loader


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data)