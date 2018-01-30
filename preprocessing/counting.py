#coding=utf-8

file = open('train_20171215.txt', 'r')
out_file = open('res.txt', 'w')
my_dict = {}
count = 1
my_dict[count] = 0
for line in file:
    list = line.split()
    print list
    if int(list[0]) == count:
        my_dict[count] += int(list[3])
    else:
        count += 1
        my_dict[count] = 0
        my_dict[count] += int(list[3])
maxnum = 0
for key, value in my_dict.iteritems():
    if value > maxnum:
        maxnum = value
    out_file.write(str(value) + '\n')

print maxnum
