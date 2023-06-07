from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation

class BuildModel:

    def __init__(self, img_shape, num_classes):

        self.img_shape = img_shape
        self.num_classes = num_classes

    def build_model(self):

        inputs = Input(shape = self.img_shape)

        down0 = Conv2D(64, (3, 3), padding = 'same')(inputs)
        down0 = Activation('relu')(down0)
        down0 = Conv2D(64, (3, 3), padding = 'same')(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides = (2, 2))(down0)

        down1 = Conv2D(128, (3, 3), padding = 'same')(down0_pool)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(128, (3, 3), padding = 'same')(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        center = Conv2D(256, (3, 3), padding='same')(down1_pool)
        center = Activation('relu')(center)
        center = Conv2D(256, (3, 3), padding='same')(center)
        center = Activation('relu')(center)

        up1 = UpSampling2D((2,2))(center)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(128, (3, 3), padding='same')(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(128, (3, 3), padding='same')(up1)
        up1 = Activation('relu')(up1)

        up0 = UpSampling2D((2,2))(up1)

        classify = Conv2D(self.num_classes, (1, 1), activation='sigmoid')(up0)

        model = Model(inputs = inputs, outputs = classify)

        return model