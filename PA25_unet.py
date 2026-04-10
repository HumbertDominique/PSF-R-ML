import tensorflow as tf
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate ,BatchNormalization, ReLU, Lambda, UpSampling2D
from tensorflow.keras.models import Model 

# Define the Dice coefficient metric for evaluating segmentation performance 
def dice_coef(y_true, y_pred): 
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    intersection = K.sum(y_true_f * y_pred_f) 

    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1) 
 

def mse_loss(y_true, y_pred): 
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

def double_conv(x, out_channels):
    x = Conv2D(out_channels, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(out_channels, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def inconv(x, out_channels):
    return double_conv(x, out_channels)

def down(x, out_channels):
    x = MaxPooling2D(2)(x)
    x = double_conv(x, out_channels)
    return x

def up(x, skip, out_channels, bilinear=True):
    if bilinear:
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    else:
        x = Conv2DTranspose(out_channels, 2, strides=2)(x)

    #Handle cropping/padding to match shape
    def crop_to_match(x, skip):
        # Crop or pad `x` to match `skip`
        x_shape = tf.shape(x)
        skip_shape = tf.shape(skip)
        height_diff = skip_shape[1] - x_shape[1]
        width_diff = skip_shape[2] - x_shape[2]
        x = tf.pad(x, [
            [0, 0],
            [height_diff // 2, height_diff - height_diff // 2],
            [width_diff // 2, width_diff - width_diff // 2],
            [0, 0]
        ])
        return x
    x = Lambda(lambda inputs: crop_to_match(inputs[0], inputs[1]))([x, skip])
    x = Concatenate()([skip, x])
    x = double_conv(x, out_channels)
    return x

def outconv(x, out_channels):
    return Conv2D(out_channels, 3, padding='same')(x)

def build_unet(input_shape=(128, 128, 2), n_channels_out=1, bilinear=True):
    inputs = Input(shape=input_shape)

    x1 = inconv(inputs, 64)
    x2 = down(x1, 128)
    x3 = down(x2, 256)
    x4 = down(x3, 512)
    x5 = down(x4, 512)

    x = up(x5, x4, 256, bilinear)
    x = up(x, x3, 128, bilinear)
    x = up(x, x2, 64, bilinear)
    x = up(x, x1, 64, bilinear)

    outputs = outconv(x, n_channels_out)

    model = Model(inputs, outputs)
    return model
