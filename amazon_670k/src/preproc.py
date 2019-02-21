count = 0

with open('../data/amazon_test.txt','r',encoding='utf-8') as fr:
    with open('../data/test.txt','w',encoding='utf-8') as fw:
        for line in fr:
            if count==0:
                count+=1
                continue
            else:
                print(line.strip(),file=fw)


count = 0

with open('../data/amazon_train.txt','r',encoding='utf-8') as fr:
    with open('../data/train.txt','w',encoding='utf-8') as fw:
        for line in fr:
            if count==0:
                count+=1
                continue
            else:
                print(line.strip(),file=fw)

