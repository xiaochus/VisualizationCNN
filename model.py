# -*- coding: utf-8 -*-
import keras.applications as app


def get_model(name='vgg16'):
    if name == 'vgg16':
        model = app.vgg16.VGG16(weights='imagenet')
        preprocess_input = app.vgg16.preprocess_input
    if name == 'vgg19':
        model = app.vgg19.VGG19(weights='imagenet')
        preprocess_input = app.vgg19.preprocess_input
    if name == 'resnet50':
        model = app.resnet50.ResNet50(weights='imagenet')
        preprocess_input = app.resnet50.preprocess_input
    if name == 'inception_v3':
        model = app.inception_v3.InceptionV3(weights='imagenet')
        preprocess_input = app.inception_v3.preprocess_input
    if name == 'xception':
        model = app.xception.Xception(weights='imagenet')
        preprocess_input = app.xception.preprocess_input
    if name == 'mobilenet':
        model = app.mobilenet.MobileNet(weights='imagenet')
        preprocess_input = app.mobilenet.preprocess_input
    if name == 'densenet':
        model = app.densenet.DenseNet121(weights='imagenet')
        preprocess_input = app.densenet.preprocess_input

    return model, preprocess_input
