# -*- coding: utf-8 -*-

def squeeze_excitation_layer(x, out_dim, ratio=16):
    '''
        SE channel attention
        input:
            x
            out_dim : channel default
            ratio : reduction rate, defualt 16
    '''
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(out_dim//ratio, activation='relu')(squeeze)
    excitation = layers.Dense(out_dim, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, out_dim))(excitation)

    scale = layers.multiply([x, excitation])

    return scale
