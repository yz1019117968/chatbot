
import re
import sys
import pickle
from tqdm import tqdm
from word_sequence import WordSequence
import numpy as np





def regular(sen):
    # sen = "".join(sen)
    if len(re.findall(r'[a-zA-Z]',sen))>0:
         return False
    return True

fp = open("xiaohuangji50w_fenciA.conv", 'r', errors='ignore', encoding='utf-8')
groups = []
group = []
for line in tqdm(fp):
    # 对于每组对话（有多行M），这些对话放在一个list中，通过group存储
    if line.startswith('M '):
        line = line.replace('\n', '')

        if '/' in line:
            # start from the first character, because the first two characters are "M" and " "
            line = line[2:].split('/')
        else:
            line = list(line[2:])
        # delete the special characters
        line = re.sub(r'[\*@#￥$%&（）\-—+=]{1,100}', '', ''.join(line))
        line = line.replace(' ','')
        line = line.replace('41：','',3)
        # if the line includes english letters, do not put it into group
        # at the same time, the sentence not only includes meaningless symbols
        if regular(line) and len(line) > len(re.findall(r'[0-9。，！？；：‘“}{】【|·~\n]',line)) and line:
            group.append(list(line))
    # 遇到E时，是每组对话的间隔，此时如果之前group中有存储上一组对话，则加入groups
    else:
        # print(group)
        if len(group) == 2:
            groups.append(group)
            group = []
        group = []
# write the regularized text into .txt file
# with open('output_regulared.txt','w',encoding='utf-8') as f:
#     for group in groups:
#         for line in group:
#             print(line)
#             # print(type(line))
#             f.write("/".join(line))
#             # f.write('\n')
#             f.write("\n")
#         f.write("\n")


# 如果文件末尾缺少了E，依然可以将最后一组对话加入groups
data_x = []
data_y = []
ws_input = WordSequence()
ws_input.fit(groups)
for group in range(len(groups)):
    for line in range(len(groups[group])):
        groups[group][line] = ws_input.transform(groups[group][line])

for group in groups:
    data_x.append(np.append(group[0],WordSequence.END))
    data_y.append(np.append(group[1],WordSequence.END))
# print(ws_input.dict)
print(data_x[0:5])
print(data_y[0:5])

pickle.dump(
    (data_x, data_y,ws_input),
    open('xiaohuangji_new.pkl', 'wb')
)


