from keras import models
from keras import layers
import numpy as np
import tensorflow as tf

class BRC(object):

    def __init__(self):
        self.model = self.create_model()

    def create_1x1(self,filters,num,tensor):
        x = layers.Conv2D(filters=filters,kernel_size=(1,1),strides=(1,1),name = 'conv'+str(num),
                          padding='same',use_bias=False)(tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def create_3x3(self,filters,num,tensor):
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), name='conv' + str(num),
                          padding='same', use_bias=False)(tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def create_layer_down(self,filters,pool_size,num,tensor):
        x = layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),name = 'conv'+str(num),
                          padding='same',use_bias=False,
                          kernel_initializer='random_uniform')(tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPool2D(pool_size=pool_size)(x)
        return x



    def create_model(self):
        """
        Model:
        :return:
        """
        # Input
        Input = layers.Input(shape=(256,256,3))

        l1 = self.create_layer_down(32,(2,2),1,Input)

        l2 = self.create_layer_down(64,(2,2),2,l1)

        l3 = self.create_layer_down(128,(2,2),3,l2)

        l4 = self.create_layer_down(256,(2,2),4,l3)

        l5 = self.create_layer_down(512,(2,2),5,l4)

        l6 = layers.Dense(512,name = 'Dense_6')(l5)
        l6 = layers.LeakyReLU(alpha=0.1)(l6)

        l7 = layers.Dense(64,name='Dense_7')(l6)
        l7 = layers.LeakyReLU(alpha=0.1)(l7)
        
        Out = layers.Dense(2,name='output')(l7)
        
        model = models.Model(Input,Out)
        return model



