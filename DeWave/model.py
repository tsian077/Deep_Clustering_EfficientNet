'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
import tensorflow as tf
import math
from constant import *



def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)
def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
# class SEBlock(tf.keras.layers.Layer):
#     def __init__(self, input_channels, ratio=0.25):
#         super(SEBlock, self).__init__()
#         self.num_reduced_filters = max(1, int(input_channels * ratio))
#         self.pool = tf.keras.layers.GlobalAveragePooling1D()
#         self.reduce_conv = tf.keras.layers.Conv1D(filters=self.num_reduced_filters,
#                                                   kernel_size=1,
#                                                   strides=1,
#                                                   padding="same")
#         self.expand_conv = tf.keras.layers.Conv1D(filters=input_channels,
#                                                   kernel_size=1,
#                                                   strides=1,
#                                                   padding="same")

#     def call(self, inputs, **kwargs):
#         branch = self.pool(inputs)
#         branch = tf.expand_dims(input=branch, axis=1)
#         branch = tf.expand_dims(input=branch, axis=1)
#         branch = self.reduce_conv(branch)
#         branch = swish(branch)
#         branch = self.expand_conv(branch)
#         branch = tf.nn.sigmoid(branch)
#         output = inputs * branch
#         return output
def SEBlock(input_tensor,input_channels, ratio=0.25):
    x = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    x = tf.expand_dims(input=x, axis=1)
    x = tf.expand_dims(input=x, axis=1)
    x =  tf.keras.layers.Conv1D(filters=max(1, int(input_channels * ratio)),
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding="same")(x)
    x  = tf.nn.swish(x)
    x = tf.keras.layers.Conv1D(filters=input_channels,
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding="same")(x)
    x = tf.nn.sigmoid(branch)
    output = input_tensor * x
    return output
    


# class MBConv(tf.keras.layers.Layer):
#     def __init__(self,input_tensor, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
#         super(MBConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.drop_connect_rate = drop_connect_rate
#         self.conv1 = tf.keras.layers.Conv1D(filters=in_channels * expansion_factor,
#                                             kernel_size=1,
#                                             strides=1,
#                                             padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         # self.dwconv = tf.keras.layers.DepthwiseConv1D(kernel_size=k,
#         #                                               strides=stride,
#         #                                               padding="same")
#         self.dwconv = tf.keras.layers.SeparableConv1D(filters=in_channels *expansion_factor,
#                                                       kernel_size=k,
#                                                       strides=stride,
#                                                       padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.se = SEBlock(input_channels=in_channels * expansion_factor)
#         self.conv2 = tf.keras.layers.Conv1D(filters=out_channels,
#                                             kernel_size=1,
#                                             strides=1,
#                                             padding="same")
#         self.bn3 = tf.keras.layers.BatchNormalization()
#         self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)
#         print('hi',self.dropout)

#     def call(self, inputs, training=None, **kwargs):
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = swish(x)
#         x = self.dwconv(x)
#         x = self.bn2(x, training=training)
#         x = self.se(x)
#         x = swish(x)
#         x = self.conv2(x)
#         x = self.bn3(x, training=training)
#         print('x_ji',x)
#         if self.stride == 1 and self.in_channels == self.out_channels:
#             if self.drop_connect_rate:
#                 x = self.dropout(x, training=training)
#             x = tf.keras.layers.add([x, inputs])
        
#         return x
def MBConv(input_tensor, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
    x = tf.keras.layers.Conv1D(filters=in_channels * expansion_factor,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same")(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.SeparableConv1D(filters=in_channels *expansion_factor,
                                                      kernel_size=k,
                                                      strides=stride,
                                                      padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = SEBlock(input_tensor=x,input_channels=in_channels * expansion_factor)
    x = tf.nn.swish(x)
    x = tf.keras.layers.Conv1D(filters=out_channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and in_channels == out_channels:

        if drop_connect_rate:
            x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x)
        x = tf.keras.layers.add([x, input_tensor])
    print('hi1111',x)
    return x

            
    

def build_mbconv_block(input_tensor,in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    print('layer_number',layers)
    for i in range(layers):
        if i==0:
            print('layer1~~')
            block = MBConv(input_tensor=input_tensor,in_channels=in_channels,
                            out_channels=out_channels,
                            expansion_factor=expansion_factor,
                            stride=stride,
                            k=k,
                            drop_connect_rate=drop_connect_rate)
        else:
            print('layer2~~')
            block = MBConv(input_tensor=block,in_channels=out_channels,
                            out_channels=out_channels,
                            expansion_factor=expansion_factor,
                            stride=1,
                            k=k,
                            drop_connect_rate=drop_connect_rate)
    return block

            
    # block = tf.keras.Sequential()
    # for i in range(layers):
    #     if i == 0:
    #         block.add(MBConv(in_channels=in_channels,
    #                          out_channels=out_channels,
    #                          expansion_factor=expansion_factor,
    #                          stride=stride,
    #                          k=k,
    #                          drop_connect_rate=drop_connect_rate))
    #     else:
    #         block.add(MBConv(in_channels=out_channels,
    #                          out_channels=out_channels,
    #                          expansion_factor=expansion_factor,
    #                          stride=1,
    #                          k=k,
    #                          drop_connect_rate=drop_connect_rate))
    # return block



class Model(object):
    def __init__(self, n_hidden, batch_size, p_keep_ff, p_keep_rc):
        '''n_hidden: number of hidden states
           p_keep_ff: forward keep probability
           p_keep_rc: recurrent keep probability'''
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.p_keep_ff = p_keep_ff
        self.p_keep_rc = p_keep_rc
        self.width_coefficient=1.0
        self.depth_coefficient=1.0
        self.dropout_rate=224
        self.drop_connect_rate=0.2
        # biases and weights for the last layer
        self.weights = {
            'out': tf.Variable(
                tf.random_normal([2 * n_hidden, EMBBEDDING_D * NEFF]))
        }
        self.biases = {
            'out': tf.Variable(
                tf.random_normal([EMBBEDDING_D * NEFF]))
        }

    def inference(self, x):
        print('---------------------------------------------')
        print('x',x)
        '''The structure of the network'''
        # four layer of LSTM cell blocks
        with tf.variable_scope('BLSTM1') as scope:
            
            lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            print('BLSTM1:',lstm_fw_cell)
            # lstm_fw_cell = tf.contrib.layers.conv1d()
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            print('BLSTM2:',lstm_fw_cell)
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate = tf.concat(outputs, 2)
        print('state_concate',state_concate)
        with tf.variable_scope('BLSTM2') as scope:
            # lstm_fw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell2, lstm_bw_cell2, state_concate,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate2 = tf.concat(outputs2, 2)
        print('state_concate2',state_concate2)
        with tf.variable_scope('BLSTM3') as scope:
            lstm_fw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell3, lstm_bw_cell3, state_concate2,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate3 = tf.concat(outputs3, 2)
        print('state_concate3',state_concate3)
        with tf.variable_scope('BLSTM4') as scope:
            lstm_fw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs4, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell4, lstm_bw_cell4, state_concate3,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate4 = tf.concat(outputs4, 2)
        # one layer of embedding output with tanh activation function
        print('state_concats4',state_concate4)
        out_concate = tf.reshape(state_concate4, [-1, self.n_hidden * 2])
        print('out_concate',out_concate)
        emb_out = tf.matmul(out_concate,
                            self.weights['out']) + self.biases['out']
        print('emb_out',emb_out)
        emb_out = tf.nn.tanh(emb_out)
        print('emb_out_tanh',emb_out)
        reshaped_emb = tf.reshape(emb_out, [-1, NEFF, EMBBEDDING_D])
        print('reshaped_emb',reshaped_emb)
        # normalization before output
        normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)
        print('normalized_emb',normalized_emb)
        print('-----------------------------------------------------------')
        return normalized_emb
# x Tensor("Placeholder_2:0", shape=(128, 100, 129), dtype=float32)
# BLSTM1: <tensorflow.contrib.rnn.python.ops.rnn_cell.LayerNormBasicLSTMCell object at 0x12c664fd0>
# BLSTM2: <tensorflow.python.ops.rnn_cell_impl.DropoutWrapper object at 0x1344b70d0>
# state_concate Tensor("BLSTM1/concat:0", shape=(128, 100, 600), dtype=float32)
# state_concate2 Tensor("BLSTM2/concat:0", shape=(128, 100, 600), dtype=float32)
# state_concate3 Tensor("BLSTM3/concat:0", shape=(128, 100, 600), dtype=float32)
# state_concats4 Tensor("BLSTM4/concat:0", shape=(128, 100, 600), dtype=float32)
# out_concate Tensor("Reshape:0", shape=(12800, 600), dtype=float32)
# emb_out Tensor("add:0", shape=(12800, 5160), dtype=float32)
# emb_out_tanh Tensor("Tanh:0", shape=(12800, 5160), dtype=float32)
# reshaped_emb Tensor("Reshape_1:0", shape=(12800, 129, 40), dtype=float32)
# normalized_emb Tensor("l2_normalize:0", shape=(12800, 129, 40), dtype=float32)
    def inference2(self,x):
        print('x',x)
        with tf.variable_scope('B0_efficientnet') as scope:
            conv1 = tf.keras.layers.Conv1D(filters=round_filters(100,self.width_coefficient),kernel_size=3,strides=1,padding='same')(x)
            print(conv1)
            bn1 = tf.keras.layers.BatchNormalization()(conv1)
            print(bn1)
            swish = tf.nn.swish(bn1)
           
            block1 = build_mbconv_block(input_tensor=swish,in_channels=round_filters(100,self.width_coefficient),
                                    out_channels=round_filters(200,self.width_coefficient),
                                    layers=round_repeats(1,self.depth_coefficient), stride=1,
                                    expansion_factor=1,k=3,drop_connect_rate=self.dropout_rate)
            
            block2 = build_mbconv_block(input_tensor=block1,in_channels=round_filters(200, self.width_coefficient),
                                        out_channels=round_filters(150, self.width_coefficient),
                                        layers=round_repeats(2, self.depth_coefficient),
                                        stride=1,
                                        expansion_factor=6, k=3, drop_connect_rate=self.drop_connect_rate)
            block3 = build_mbconv_block(input_tensor=block2,in_channels=round_filters(150, self.width_coefficient),
                                        out_channels=round_filters(300, self.width_coefficient),
                                        layers=round_repeats(2, self.depth_coefficient),
                                        stride=1,
                                        expansion_factor=6, k=5, drop_connect_rate=self.drop_connect_rate)
            block4 = build_mbconv_block(input_tensor=block3,in_channels=round_filters(300, self.width_coefficient),
                                         out_channels=round_filters(250, self.width_coefficient),
                                         layers=round_repeats(3, self.depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=self.drop_connect_rate)
            block5 = build_mbconv_block(input_tensor=block4,in_channels=round_filters(250, self.width_coefficient),
                                         out_channels=round_filters(400, self.width_coefficient),
                                         layers=round_repeats(3, self.depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=self.drop_connect_rate)
            block6 = build_mbconv_block(input_tensor=block5,in_channels=round_filters(400, self.width_coefficient),
                                         out_channels=round_filters(350, self.width_coefficient),
                                         layers=round_repeats(4, self.depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=self.drop_connect_rate)
            block7 = build_mbconv_block(input_tensor=block6,in_channels=round_filters(350, self.width_coefficient),
                                         out_channels=round_filters(600, self.width_coefficient),
                                         layers=round_repeats(1, self.depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

            out_concate = tf.reshape(block4, [-1, self.n_hidden * 2])
            print(out_concate)
            emb_out = tf.matmul(out_concate,
                            self.weights['out']) + self.biases['out']
            print(emb_out)
            emb_out = tf.nn.tanh(emb_out)
            print(emb_out)
            reshaped_emb = tf.reshape(emb_out, [-1, NEFF, EMBBEDDING_D])
            print(reshaped_emb)
            normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)
            print(normalized_emb)

            return normalized_emb
            
           
            
                                    
            
            
            


    def loss(self, embeddings, Y, VAD):
        '''Defining the loss function'''
        embeddings_rs = tf.reshape(embeddings, shape=[-1, EMBBEDDING_D])
        VAD_rs = tf.reshape(VAD, shape=[-1])
        # get the embeddings with active VAD
        embeddings_rsv = tf.transpose(
            tf.multiply(tf.transpose(embeddings_rs), VAD_rs))
        embeddings_v = tf.reshape(
            embeddings_rsv, [-1, FRAMES_PER_SAMPLE * NEFF, EMBBEDDING_D])
        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_rsv = tf.transpose(
            tf.multiply(tf.transpose(Y_rs), VAD_rs))
        Y_v = tf.reshape(Y_rsv, shape=[-1, FRAMES_PER_SAMPLE * NEFF, 2])
        # fast computation format of the embedding loss function
        loss_batch = tf.nn.l2_loss(
            tf.matmul(tf.transpose(embeddings_v, [0, 2, 1]), embeddings_v)) -  \
            2 * tf.nn.l2_loss(tf.matmul(tf.transpose(embeddings_v, [0, 2, 1]), Y_v)) + \
            tf.nn.l2_loss(tf.matmul(tf.transpose(Y_v, [0, 2, 1]), Y_v))
        loss_v = (loss_batch) / self.batch_size / (FRAMES_PER_SAMPLE^2)
        tf.summary.scalar('loss', loss_v)
        return loss_v

    def train(self, loss, lr):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(
            zip(gradients, v))
        return train_op
