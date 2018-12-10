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


    def create_layer_up(self,filters,up_size,num,tensor):
        x = layers.UpSampling2D(size = up_size,name = 'upsample'+str(num))(tensor)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), name='conv' + str(num),
                          padding='same', use_bias=False,
                          kernel_initializer='random_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
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

        l4 = self.create_layer_down(128,(2,2),4,l3)

        skip_connection = self.create_1x1(64,8,l4)

        l5 = self.create_1x1(256,5,l4)

        l6 = self.create_1x1(512,6,l5)

        l7 = self.create_1x1(1024,7,l6)

        l8 = layers.Concatenate(axis=3)([skip_connection,l7])

        l9 = self.create_1x1(1024,9,l8)

        l10 = self.create_layer_down(1024,(2,2),10,l9)

        l11 = self.create_1x1(1024,11,l10)

        l12 = layers.Dense(512,activation='sigmoid', name='Dense_12')(l11)

        l13 = self.create_layer_down(512,(2,2),13,l12)

        l14 = self.create_layer_down(512,(2,2),14,l13)

        l15 = self.create_layer_down(512,(2,2),15,l14)

        l16 = layers.Dense(256,activation='sigmoid',name = 'Dense_16')(l15)

        l17 = layers.Dense(64,activation='sigmoid',name='Dense_17')(l16)

        Out = layers.Dense(2,activation='sigmoid',name='output')(l17)
        Out = layers.Lambda(lambda x: np.subtract(np.multiply(x,5.0),
                                                  tf.convert_to_tensor(np.array([[[0,2.5]]]).astype(np.float32))))(Out)

        model = models.Model(Input,Out)
        return model




