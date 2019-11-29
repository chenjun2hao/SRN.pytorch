import json

path = '/home/deepblue/deepbluetwo/chenjun/data/train_task2_labels.json'
with open(path, 'r') as f:
    datas = json.load(f)

outf = open('./dataset/ICDAR2019/icdar2019_train.txt', 'w')

k = 0
for key, value in datas.items():
    imgname = key
    label = value[0]['transcription']
    outf.write(f'{imgname}.jpg\t{label}\n')
    outf.flush()
    k += 1
    print(k)

outf.close()