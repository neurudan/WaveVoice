import os

path = 'dataset/TIMIT'

train = []
test = []
for f in os.listdir(path+'/TRAIN'):
    train.extend(os.listdir(path+'/TRAIN/'+f))
for f in os.listdir(path+'/TEST'):
    test.extend(os.listdir(path+'/TEST/'+f))

train.remove('.DS_Store')
train.remove('._.DS_Store')
test.remove('.DS_Store')
test.remove('._.DS_Store')
print(len(train))
print(len(test))
train.extend(test)
train.remove('.DS_Store')
train.remove('._.DS_Store')
print(len(list(set(train))))
d = {9: [], 5:[], 11:[]}
for t in train:
    d[len(t)].append(t)
print(len(d[5]))
print(len(d[9]))
print(len(d[11]))
print()
print(d[9])
print(d[11])
