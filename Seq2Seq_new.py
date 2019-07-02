import numpy as np
import tensorflow as tf
from tensorflow import layers, Tensor
from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper
import data_utils
from word_sequence import WordSequence
from tqdm import tqdm
import pickle
from flask import Flask,request
from nltk.translate.bleu_score import corpus_bleu
class Seq2Seq():
    def __init__(self,input_vocab_size,hidden_units,depth,learning_rate,input_size,wordSequence,mode,use_bidirection=False,beam_width=None,use_dropout=False,seed=2,keep_prob_holder=0.9,use_residual=False,embedding_size = 128):

        self.use_bidirection = use_bidirection
        self.initializer = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32)
        self.global_step = tf.Variable(
            0, trainable=False, name='global_step'
        )
        self.input_size = input_size  # 输入的行数,其实是batch_size
        self.mode = mode  # two modes: train or inference
        print("current mode: {}".format(self.mode))
        self.use_dropout = use_dropout
        self.seed = seed
        self.keep_prob_holder = keep_prob_holder
        self.use_residual = use_residual
        self.hidden_units = hidden_units
        self.depth = depth  # 网络深度
        self.input_vocab_size = input_vocab_size  # 字典大小
        self.embedding_size = embedding_size  # 单个字embedding后的维度
        self.learning_rate = learning_rate
        # self.decay_step = decay_step
        self.beam_width = beam_width
        self.wordSequence = wordSequence
    def init_placeholder(self):
        self.init_placeholder_encoder()
        self.init_placeholder_decoder()
    def init_placeholder_encoder(self):
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.input_size,None),
            name='encoder_input'
        )
        # [20,30,5,7.....]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.input_size,),
            name='encoder_input_length'
        )

    def init_placeholder_decoder(self):
        if self.mode =='train':
            self.decoder_inputs = tf.placeholder(
                    dtype=tf.int32,
                    shape=(self.input_size,None),
                    name='decoder_inputs'
                )

            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=(self.input_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape= (self.input_size,1),
                dtype= tf.int32
            ) * WordSequence.START
            self.decoder_inputs_train = tf.concat([self.decoder_start_token,
                                                   self.decoder_inputs],axis=1)
    def build_single_cell(self):
        cell = BasicLSTMCell(self.hidden_units,state_is_tuple=True)
        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob= self.keep_prob_holder,
                seed=self.seed
            )

        if self.use_residual:
            # 默认是输入与输出相加
            cell = ResidualWrapper(cell)

        return cell

    def build_cell(self):
        return MultiRNNCell([
            self.build_single_cell()
            for _ in range(self.depth)])
    def trans_sentence(self,result):
        return  self.wordSequence.inverse_transform(result)
    def build_model(self):
        common_embeddings = tf.get_variable(
            name='embeddings',
            shape=(self.input_vocab_size, self.embedding_size),
            initializer=self.initializer,
            dtype=tf.float32,
            trainable=True
        )
        # encoder
        encoder_inputs_embedded = tf.nn.embedding_lookup(
                                            params=common_embeddings,
                                            ids=self.encoder_inputs
                                        )
        if self.use_residual:
            encoder_inputs_embedded = layers.dense(inputs=encoder_inputs_embedded,units=self.hidden_units,use_bias=False)
        encoder_cell = self.build_cell()
        if not self.use_bidirection:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs_embedded,
                                                           sequence_length=self.encoder_inputs_length,
                                                           dtype=tf.float32)
        else:
            encoder_cell_bw = self.build_cell()
            (
                (encoder_fw_outputs, encoder_bw_outputs),
                (encoder_fw_state, encoder_bw_state)
            ) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell,
                cell_bw=encoder_cell_bw,
                inputs=encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32,
            )

            # 首先合并两个方向 RNN 的输出
            encoder_outputs = tf.concat(
                (encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_state = []
            for i in range(self.depth):
                encoder_state.append(encoder_fw_state[i])
                encoder_state.append(encoder_bw_state[i])
            encoder_state = tuple(encoder_state)
            encoder_state = encoder_state[-self.depth:]
        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.input_size
        # decoder
        if self.mode == 'inference':
            print("inference setting:")
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            # 如果使用了 beamsearch 那么输入应该是 beam_width 倍于 batch_size 的
            batch_size *= self.beam_width

        attention = BahdanauAttention(
            num_units=self.hidden_units,
            memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length
        )
        decoder_cell = self.build_cell()

        def cell_input_fn(inputs, attention):
            """根据attn_input_feeding属性来判断是否在attention计算前进行一次投影计算
            """
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))
        decoder_cell = AttentionWrapper(
            cell=decoder_cell,
            attention_mechanism=attention,
            attention_layer_size=self.hidden_units,
            cell_input_fn=cell_input_fn
        )

        decoder_init_state = decoder_cell.zero_state(
            batch_size=batch_size, dtype=tf.float32
        )
        # 使用encoder输出的state
        decoder_init_state = decoder_init_state.clone(
            cell_state=encoder_state
        )
        # print(decoder_init_state)
        # fully connection
        full_conn = layers.Dense(
            self.input_vocab_size,
            dtype=tf.float32,
            use_bias=True,
            name='fully_connected_layer'
        )
        if self.mode == 'train':
            print("train setting:")
            # initiate input of decoder
            dencoder_inputs_embedded = tf.nn.embedding_lookup(
                params=common_embeddings,
                ids=self.decoder_inputs_train
            )
            training_helper = seq2seq.TrainingHelper(
                inputs=dencoder_inputs_embedded,
                sequence_length=self.decoder_inputs_length,
                name='training_helper'
            )
            # 此处已经为decoder添加了最后的dense layer
            # 后面final_output是经过dense layer的输出
            training_decoder = seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_init_state,
                output_layer=full_conn
            )
            # 句子中的最大长度，也就是最大时间步数
            max_decoder_length = tf.reduce_max(
                self.decoder_inputs_length
            )

            # decoder_output <<<<<<<<<############################################################################
            final_output, final_state, _ = seq2seq.dynamic_decode(
                decoder=training_decoder,
                maximum_iterations=max_decoder_length
            )
            decoder_logits_train = final_output.rnn_output
            # decoder_label = tf.argmax(decoder_logits_train,1)
            # 一个batch中每一个字的损失
            train_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_inputs,
                logits=decoder_logits_train)
            # 指明batch中，每句话的实际语句部分和补位部分
            masks = tf.sequence_mask(
                lengths=self.decoder_inputs_length,
                maxlen=max_decoder_length,
                dtype=tf.float32,
                name='mask'
            )
            # 一个batch的求和loss
            self.loss = seq2seq.sequence_loss(
                logits=decoder_logits_train,
                targets=self.decoder_inputs,
                weights=masks
            )
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=tf.trainable_variables())
        if self.mode == 'inference':
            print("inference setting2:")
            start_tokens = tf.tile(
                [WordSequence.START],
                [self.input_size]
            )
            end_token = WordSequence.END

            def embed_and_input_proj(inputs):
                """输入层的投影层wrapper
                """
                return tf.nn.embedding_lookup(
                    common_embeddings,
                    inputs
                )

            inference_decoder = BeamSearchDecoder(
                cell=decoder_cell,
                embedding=embed_and_input_proj,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=decoder_init_state,
                beam_width=self.beam_width,
                output_layer=full_conn,
            )
            # max_decode_step = tf.round(tf.reduce_max(
            #     self.encoder_inputs_length))

            (
                decoder_outputs_decode,
                final_state,
                _  # self.decoder_outputs_length_decode
            ) = (seq2seq.dynamic_decode(
                decoder=inference_decoder,
                # impute_finished=True,	# error occurs
                # maximum_iterations=max_decode_step,
                swap_memory=True
            ))
            self.result = decoder_outputs_decode.predicted_ids[:,:,0]
            self.pred_score = decoder_outputs_decode.beam_search_decoder_output.scores
            # print("result {}".format(self.result))





# input = np.array([[8,9,7,4,3,0,0],[5,9,6,6,7,3,0],[8,6,4,6,6,8,3]])
# output = np.array([[4,5,7,8,3,0,0],[4,5,6,6,7,3,0],[4,6,6,6,8,8,3]])
# input_length = np.array([5,6,7])
# output_length = np.array([5,6,7])

# initialization








#
# def load(sess, save_path='model.ckpt'):
#     """读取模型"""
#     print('try load model from', save_path)
#     tf.train.Saver().saver.restore(sess, save_path)




def train():
    from matplotlib import pyplot as plt
    # initialization
    n_epoch = 1
    batch_size = 64
    x_data, y_data,ws = pickle.load(open('xiaohuangji_new.pkl', 'rb'))
    SeqToSeq = Seq2Seq(input_vocab_size=len(ws.dict), hidden_units=256, depth=4, mode='train', learning_rate=0.001,
                       input_size=batch_size,wordSequence=ws,use_dropout=True,use_residual=True,use_bidirection=True,
                       embedding_size=256)
    # embedding matrix
    SeqToSeq.init_placeholder()
    SeqToSeq.build_model()
    config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    init = tf.initialize_all_variables()
    with tf.Session(config=config) as sess:
        sess.run(init)
        steps = int(len(x_data) / batch_size) + 1
        for epoch in range(1, n_epoch + 1):
            # losses = []
            bar = tqdm(range(steps), total=steps,
                       desc='epoch {}, loss=0.000000'.format(epoch))
            x=[]
            y=[]
            for i in bar:
                x.append(i)
                x_batch, y_batch, x_lens, y_lens = data_utils.get_batch(x_data, y_data, batch_size)
                loss,_= sess.run([SeqToSeq.loss,SeqToSeq.train_op],feed_dict={SeqToSeq.encoder_inputs: x_batch,
                                                               SeqToSeq.encoder_inputs_length: x_lens,SeqToSeq.decoder_inputs:y_batch,SeqToSeq.decoder_inputs_length:y_lens})
                y.append(loss)
                # losses.append(loss)
                bar.set_description('epoch {} loss={:.6f}'.format(
                    epoch,
                    loss
                ))
            plt.plot(x,y)
            plt.savefig("./figures/epoch {}.png".format(epoch))
            # plt.show()
        # save trained model
        tf.train.Saver().save(sess, save_path='./model/chatbot.ckpt')
        print("model saved")

def predict():
    import extractResponses as er
    save_path = './model/chatbot.ckpt'
    x_data, y_data, ws = pickle.load(open('xiaohuangji_new.pkl', 'rb'))
    SeqToSeq = Seq2Seq(input_vocab_size=len(ws.dict), hidden_units=256, depth=4, mode='inference', learning_rate=0.001,
                       input_size=5,wordSequence=ws,beam_width=3,use_dropout=True,use_residual=True,use_bidirection=True,
                       embedding_size=256)
    SeqToSeq.init_placeholder_encoder()
    SeqToSeq.build_model()


    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    with tf.Session(config=config) as sess:
        tf.train.Saver().restore(sess, save_path)
        # x_batch, _, x_lens, _ = data_utils.get_batch(x_data, y_data, 5)
        x_batch_c = [np.array(['你','好','呀','</s>','<pad>','<pad>','<pad>']),
                     np.array(['你','叫','什','么','名','字','</s>']),
                     np.array(['你','几','岁','了','</s>','<pad>','<pad>']),
                     np.array(['我','喜','欢','你','</s>','<pad>','<pad>']),
                     np.array(['我','想','要','女','朋','友','</s>'])]
        # reference sentences:
        references = er.extractResponses()
        x_lens = [4,7,5,5,7]
        x_batch = []
        for sentence in x_batch_c:
            sent = []
            for id in sentence:
                sent.append(ws.to_index(id))
            print(sent)
            x_batch.append(np.array(sent))
        print(x_batch,x_lens)
        result,pred_score = sess.run([SeqToSeq.result,SeqToSeq.pred_score],feed_dict={SeqToSeq.encoder_inputs:x_batch,SeqToSeq.encoder_inputs_length:x_lens})
        print(result)
        print(result.shape)
        candidates = []
        for sentence in result:
            sent = []
            for id in sentence:
                if id == 3:
                    break
                else:
                    sent.append(ws.to_word(id))
            candidates.append(sent)
            print(sent)
        print(references)
        print('Cumulative 1-gram: %f' % corpus_bleu(references, candidates, weights=(1, 0, 0, 0)))
        print('Cumulative 2-gram: %f' % corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0)))
        print('Cumulative 3-gram: %f' % corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0)))
        print('Cumulative 4-gram: %f' % corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25)))
def chatbot(infos):
    save_path = './model/chatbot.ckpt'
    x_data, y_data, ws = pickle.load(open('xiaohuangji_new.pkl', 'rb'))
    SeqToSeq = Seq2Seq(input_vocab_size=len(ws.dict), hidden_units=256, depth=4, mode='inference', learning_rate=0.001,
                       input_size=1,wordSequence=ws,beam_width=3,use_dropout=True,use_residual=True,use_bidirection=True,
                       embedding_size=256)
    SeqToSeq.init_placeholder_encoder()
    SeqToSeq.build_model()


    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    with tf.Session(config=config) as sess:
        tf.train.Saver().restore(sess, save_path)
        x_batch = list(infos)
        x_batch.append('</s>')
        sent = []
        for id in x_batch:
            sent.append(ws.to_index(id))
        x_batch = []
        x_batch.append(sent)
        # print(np.array(x_batch).shape)
        x_lens = [len(x_batch)]
        result,pred_score = sess.run([SeqToSeq.result,SeqToSeq.pred_score],feed_dict={SeqToSeq.encoder_inputs:np.array(x_batch),SeqToSeq.encoder_inputs_length:x_lens})

        final_result = []
        # print(result)
        for sentence in result:
            final_result = []
            for id in sentence:
                if id == 3:
                    break
                else:
                    final_result.append(ws.to_word(id))
        print(final_result)
        return "".join(final_result)
app = Flask(__name__)
@app.route('/api/chatbot', methods=['get'])
def ask_chatbot():
    infos = request.args['infos']
    import json
    result = chatbot(infos)
    return result

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
# if __name__ == '__main__':
#     print(chatbot("你好呀~"))























