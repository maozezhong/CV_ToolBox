# -*- coding=utf-8 -*-
def add_attention(base_model):
    '''
    输入:
        base_model:基础模型
    输出:
        加了attention后的basemodel
    '''
    ################## add attention by mao 2019-6-5 21:00 ##################
    #加attention, 点乘后求和
    def getSum(input_tensor):
        '''
        input_tensor : [None, 49, 2048]
        Note:
            函数里面要用的都得在函数内import！！！！！！
        '''
        import keras.backend as K
        res = K.sum(input_tensor, axis=-2)
        return res

    from keras.layers import multiply, Reshape, RepeatVector, Permute, Lambda, Dense, BatchNormalization
    from keras.layers.pooling import GlobalAveragePooling2D
    from keras.models import Model

    x = base_model.output
    print(x.shape)
    _, H,W,C = x.shape
    H = int(H)
    W = int(W)
    C = int(C)
    x = GlobalAveragePooling2D(name='avg_pool_for_attention')(x)    #[None, 7, 7, 2048] -> [None, 1, 1, 2048]
    x = Dense(H*W, activation='softmax', name='attention_w')(x)     #全连接层，输出系数
    x = Reshape((H*W,))(x)                                          #[None, 1, 49, 1] -> [None, 49]
    x = RepeatVector(C)(x)                                          #[None, 49] -> [None, 2048, 49]
    x = Permute((2,1))(x)                                           #[None, 2048, 49] -> [None, 49, 2048]
    x = Reshape((H, W, C))(x)                                       #[None, 49, 2048] -> [None, 7, 7, 2048]
    x = multiply([base_model.output, x])                            #逐个元素乘积
    base_model = Model(inputs=base_model.input, outputs=x)
    ############################## end of attention #########################

    return base_model