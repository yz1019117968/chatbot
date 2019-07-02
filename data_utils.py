import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import WordSequence
import pickle

# VOCAB_SIZE_THRESHOLD_CPU = 50000
# def _get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
# def _get_embed_device(vocab_size):
#     gpus = _get_available_gpus()
#     if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
#         return "/cpu:0"
#     return "/gpu:0"



# datax和datay已经分别转为index
def get_batch(data_x,data_y, batch_size):
    ps = []
    # 对一个batch的数据从源数据中随机抽样（有放回）
    while len(ps) < batch_size:
        ps.append(random.randint(0, len(data_x) - 1))

    x_batch = []
    y_batch = []

    x_lens = [len(data_x[p]) for p in ps]
    y_lens = [len(data_y[p]) for p in ps]

    max_x_len = max(x_lens)
    max_y_len = max(y_lens)

    for p in ps:
        x = np.append(data_x[p],[WordSequence.PAD]*(max_x_len-len(data_x[p])))
        y = np.append(data_y[p],[WordSequence.PAD]*(max_y_len-len(data_y[p])))
        x_batch.append(x.astype(int))
        y_batch.append(y.astype(int))

    return x_batch,y_batch,x_lens,y_lens


if __name__ == '__main__':
    x_data, y_data,_ = pickle.load(open('xiaohuangji.pkl', 'rb'))
    # print(len(x_data))
    print(get_batch(x_data, y_data,5))

