import pandas as pd
import re



#basic pattern:


#最优先：标准的"i5-4800u"模式，正则：'i\d\W\d{3,}[a-z]{,2}\W' (not used)
#包括"i5"，根据官网定义的型号(比如4700u)，以及根据观察数据的型号(4900mq)
#这部分正则包括：
#core/pentium/celeron:
#\Wi\d\W
#\W\d{3,5}[efhkmstpuyxb]{1}[kfqes]{0,1}\W
#\W\d{4,5}g\d\W
#\Wg\d{3,4}[teuy]{,2}\W
#\W[nj]\d{4}\W
#atom: \Wc[23]\d{3}[rl]{0,1}\W
#atom: \Wp59\d{2}b\W
#xeon platinum: \W83\d{2}[ynqvmh]{0,1}[l]{0,1}\W
#xeon gold: \W[56]3\d{2}[ynuth]{0,1}[l]{0,1}\W
#xeon silver: \W42\d{2}[rty]{0,1}\W -> 拓展：直接识别纯四位数字 \W\d{4}\W
#xeon bronze: \W320\d[r]{0,1}\W
#xeon D-processor: \Wd-*[12]\d{3}[nit]{0,1}[te]{0,1}
#xeon E-processor: \We-*2[123]\d{2}[gm]{0,1}[el]{0,1}[l]{0,1}\W
#xeon W-processor: \Ww-*\d{4,5}[mptx]{0,1}[elr]{0,1}[e]{0,1}\W
#duo: \W[etulx]\d{4}\W


#amd pattern
#纯数字+字母：\W\d{3,4}[wxhupkctmedb]{1}[xtse]{0,1}\W; \W\d{4}wx\W|\W\d{4}x\W|\W\d{4}h[xs]*\W|\W\d{4}xt\W|\W\d{3,4}u\W|\W\d{4}g[e]*\W|\W\d{4}c[e]*\W|\W\d{4}e\W|\W\d{3}ge\W|\W\d{4}b\W|
# 注意在这里我忽略了纯数字的情况，因为在title中出现纯数字目前无法分辨它是否是CPU型号；
#A8-数字+末尾字母类：\Wa[4689]\W|\Wa1[012]\W|\Wa[4689]-\d{4}[kpm]\W|\Wa1[012]-\d{4}[kpm]\W, 
#A8-纯数字类：\Wa[4689]-\d{4}\W|\Wa1[012]-\d{4}\W
#数字+字母+数字: \W\d{2}[f]\d{1}\W|\W7[fh]\d{2}\W
#fx系列： \Wfx[-\s]*\d{4}[ep]{0,1}\W
#he结尾：\W\d{4}[-\s]*he\W
def HierarchicalBlock(data,attr,limit):
    patterns=[]
    
    basicKeyword=['intel', 'amd','duo','celeron', 'pentium', 'centrino','xeon','platinum','atom','e-series', 'radeon', 'athlon', 'turion', 'phenom','ryzen','epyc']
    IntelPattern=['\Wi\d\W','\W\d{3,5}[efhkmsptuyxb]{1}[kfqes]{0,1}\W','\W\d{4,5}g\d\W','\Wg\d{3,4}[teuy]{,2}\W','\W[nj]\d{4}\W','\Wc[23]\d{3}[rl]{0,1}\W','\Wp59\d{2}b\W',
            '\W83\d{2}[ynqvmh]{0,1}[l]{0,1}\W','\W[56]3\d{2}[ynuth]{0,1}[l]{0,1}\W','\W\d{4}\W','\W320\d[r]{0,1}\W','\Wd-*[12]\d{3}[nit]{0,1}[te]{0,1}',
            '\We-*2[123]\d{2}[gm]{0,1}[el]{0,1}[l]{0,1}\W','\Ww-*\d{4,5}[mptx]{0,1}[elr]{0,1}[e]{0,1}\W','\W[etulx]\d{4}\W']
    amdPattern=['\W\d{4}wx\W|\W\d{4}x\W|\W\d{4}h[xs]*\W|\W\d{4}xt\W|\W\d{3,4}u\W|\W\d{4}g[e]*\W|\W\d{4}c[e]*\W|\W\d{4}e\W|\W\d{3}ge\W|\W\d{4}b\W',
            '\Wa[4689]\W|\Wa1[012]\W|\Wa[4689]-\d{4}[kpm]\W|\Wa1[012]-\d{4}[kpm]\W','\Wa[4689]-\d{4}\W|\Wa1[012]-\d{4}\W','\W\d{2}[f]\d{1}\W|\W7[fh]\d{2}\W',
            '\Wfx[-\s]*\d{4}[ep]{0,1}\W','\W\d{4}[-\s]*he\W']
    for reg in IntelPattern:
        patterns.append(re.compile(reg))
    for reg in amdPattern:
        patterns.append(re.compile(reg))
    for reg in basicKeyword:
        patterns.append(re.compile(reg))
    groups={}
    #first group by brands
    brands=['dell', 'lenovo', 'acer', 'asus', 'hp','panasonic','toshiba','xmg']
    for i in range(data.shape[0]):
        title=data[attr][i].lower()
        my_brand=[]
        for b in brands:
            if title.find(b)>=0:
                my_brand.append(b)
        string=" ".join(my_brand)
        try:
            groups[string][1].append(i)
        except:
            groups[string]=[my_brand,[i]]

    #second group by pc family
    families=['spectre','envy','pavilion','laptop','chromebook','omen','victus','touchsmart',  #hp
            'thinkpad','ideapad','thinkbook','yoga','legion', #lenovo
            'xps','latitude','inspiron','vostro','alienware','precision','rugged', 'pavillion',#dell
            'enduro','swift','spin','aspire','porsche','nitro', #acer
            'zenbook','studiobook','expertbook','zephyrus','vivobook'
            'toughbook','netbook','dynabook','tecra','elitebook','lifebook','ultrabook']
    second_groups={}
    for key in groups:
        group=groups[key]
        feature=group[0]
        ids=group[1]
        for i in ids:
            my_feature=[]
            title=data[attr][i].lower()
            my_feature+=feature
            for f in families:
                if title.find(f)>=0:
                    my_feature.append(f)
            my_feature.sort()
            string=" ".join(my_feature)
            try:
                second_groups[string][1].append(i)
            except:
                second_groups[string]=[my_feature,[i]]
    
    #third group by cpu
    third_groups={}
    for key in second_groups:
        group=second_groups[key]
        feature=group[0]
        ids=group[1]
        for i in ids:
            my_feature=[]
            title=' '+data[attr][i].lower()+' '
            my_feature+=feature
            result=set()
            for pattern in patterns:
                res=pattern.findall(title)
                for string in res:
                    splitted=re.split('\W',string)
                    for substr in splitted:
                        if len(substr)>1:
                            result.add(substr)
            my_feature+=list(result)
            my_feature.sort()
            string=" ".join(my_feature)
            try:
                third_groups[string][1].append(i)
            except:
                third_groups[string]=[my_feature,[i]]

    #find groups
    unsolved=[]
    left=[]
    right=[]
    for key in third_groups:
        group=third_groups[key][1]
        length=len(group)
        # print(key)
        # for id in group:
        #     print("%d: %s"%(data['cluster'][id],data[attr][id]))
        # print()
        if length>1:
            for i in range(length-1):
                for j in range(i+1,length):
                    id1=data['id'][group[i]]
                    id2=data['id'][group[j]]
                    if id1==id2:
                        continue
                    left.append(min(id1,id2))
                    right.append(max(id1,id2))
                    if len(left)==limit:
                        return left,right
        else:
            unsolved.append(group[0])
    left.extend([0]*(limit-len(left)))
    right.extend([0]*(limit-len(right)))    
    # if len(left)<1000000:
    #     unsolved.sort(key=lambda x:len(x[0]))
    #     maxsize=1000000-len(left)
    #     cnt=0
    #     i=1
    #     while i<len(unsolved):
    #         s1=set(unsolved[i-1][0].split(' '))
    #         s2=set(unsolved[i][0].split(' '))
    #         jaccard=len(s1.intersection(s2))/max(len(s1),len(s2))
    #         if jaccard>0.8:
    #             cnt+=1
    #             id1=data['id'][unsolved[i-1][1][0]]
    #             id2=data['id'][unsolved[i][1][0]]
    #             if id1==id2:
    #                 continue
    #             left.append(min(id1,id2))
    #             right.append(max(id1,id2))
    #             if cnt>=maxsize:
    #                 return left,right
    #             i+=2
    #         else:
    #             i+=1
    return left,right

    # for i in range(data.shape[0]):
    #     title=data[attr][i].lower()+' '
    #     result=set()
    #     for pattern in patterns:
    #         res=pattern.findall(title)
    #         for string in res:
    #             splitted=re.split('\W',string)
    #             for substr in splitted:
    #                 if len(substr)>1:
    #                     result.add(substr)
    #     result=list(result)
    #     result.sort()
    #     key=" ".join(result)
    #     try:
    #         groups[key].append(i)
    #     except:
    #         groups[key]=[i]
    

if __name__=='__main__':
    x1=pd.read_csv("X1.csv")
    left,right=HierarchicalBlock(x1,'title',1000000)
    # x2=pd.read_csv("../records/x2_sort.csv")
    # x2_left,x2_righy=HierarchicalBlock(x2,'name',2000000)
    # left+=x2_left
    # right+=x2_righy
    left.extend([0]*2000000)
    right.extend([0]*2000000)
    output=pd.DataFrame()
    output['left_instance_id']=left
    output['right_instance_id']=right
    output.to_csv("output.csv",index=False)
    
