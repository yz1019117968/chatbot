import numpy as np

import numpy as np

class WordSequence(object):

    PAD_TAG = '<pad>'
    UNK_TAG = '<unk>'
    START_TAG = '<s>'
    END_TAG = '</s>'

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        #初始化基本的字典dict
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        self.fited = False
    # 将每个字转为索引
    def to_index(self, word):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def size(self):

        assert self.fited, "WordSequence 尚未进行 fit 操作"
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    def fit(self, sentences, min_count=None, max_count=None, max_features=None):

        assert not self.fited, 'WordSequence 只能fit一次'
        count = {}
        for sentence in sentences:
            for sub_s in sentence:
                arr = list(sub_s)
                for a in arr:
                    if a not in count:
                        count[a] = 0
                    count[a] += 1
        # print(sorted(count.keys()))
        # smaller than min would be deleted
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x:x[1])
            print('count:')
            print(count)
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
                print(count)
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            # print('count1:')
            # print(count)
            # 将count中的字逐一加入字典
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        # print(self.dict)
        # build the dictionary from the inputs data
        print(self.dict)
        self.fited = True

    def transform(self, sentence, max_len=None):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        # 先全部置为pad
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)

    def inverse_transform(self, indices):
        ret = []
        for i in indices:
            if i == WordSequence.PAD:
                continue
            if i == WordSequence.UNK:
                continue
            if i == WordSequence.START:
                continue
            if i == WordSequence.END:
                continue
            word = self.to_word(i)
            ret.append(word)

        return ret
def test():

    ws = WordSequence()
    ws.fit([[
        ['你', '好', '啊'],
        ['你', '好', '哦'],
    ],[['我','话','怒','和','我'],['哥','个']]])
    indice = ws.inverse_transform([1,4,6,7,8,5])
    print(indice)

    # back = ws.inverse_transform(indice)


if __name__ == '__main__':
    test()





