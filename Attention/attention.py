# 待修改
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

    from keras.layers import merge, Reshape, RepeatVector, Permute, Lambda
    x = base_model.output
    _, H, W, C = x.shape
    x = Reshape((1, H*W, 2048))(x)   #[None, 7, 7, 2048] -> [None, 1, 49, 2048]
    x = Conv2D(1, (1,1), activation=softMaxAxis(-2), strides=(1,1), name='attention_feature')(x)    #[None, 1, 49, 2048] -> [None, 1, 49, 1]
    x = Reshape((49,))(x)           #[None, 1, 49, 1] -> [None, 49]
    x = RepeatVector(2048)(x)       #[None, 49] -> [None, 2048, 49]
    x = Permute((2,1))(x)           #[None, 2048, 49] -> [None, 49, 2048]
    x = Reshape((7, 7, 2048))(x)    #[None, 49, 2048] -> [None, 7, 7, 2048]
    x = merge([base_model.output, x], name='attention_mul', mode='mul') #点乘
    x = Reshape((49, 2048))(x)      #[None, 7, 7, 2048] -> [None, 49, 2048]
    x = Lambda(getSum)(x)           #对[49]这个位置求和并且reshape输出为[None, 2048]
    x = Reshape((1,1,2048))(x)      #[None, 2048] -> [None, 1, 1, 2048]
    base_model = Model(inputs=base_model.input, outputs=x)
    ############################## end of attention #########################

    return base_model