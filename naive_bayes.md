
## 朴素贝叶斯的实现并用于垃圾邮寄分类测试
朴素贝叶斯是基于贝叶斯定理与特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的联合概率分布，然后基于此模型，对给定的输入x,利用贝叶斯定理求出后验概率最大的输出y。
朴素贝叶斯法实际学习到生成数据的机制，即数据特征和类别的联合概率分布 P(X,Y)，所以属于生成模型。

***
## 贝叶斯公式与朴素贝叶斯模型
$$ P(Y|X) = \frac{P(Y)P(X|Y)}{P(X)} $$

通过贝叶斯公式可以看出，朴素贝叶斯法分类时，对给定的输入x,通过学习到的模型九三后验概率分布$ P(Y=c_k|X=x) $, 将后验概率最大的类作为x的类输出。

后验概率计算根据贝叶斯公式可得：

$$ P(Y=c_k|X=x)=\frac{P(Y=c_k)P(X=x|Y=c_k)}{\sum_kP(Y=c_k)P(X=x|Y=c_k)} $$

因为对于每个样本 p(x)的总是相同的，所以在不同类别的后验概率的计算中分母是相同的，可以省略掉。这样贝叶斯分类器可以表示为：

$$ y=argmax_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k) $$ 其中 j 代表X特征个数，k代表类别的个数

***
## 后验概率最大化的含义
在朴素贝叶斯中，将实例分到后验概率最大的类中，这等价于期望风险最小化。

***
## 朴素贝叶斯法的参数估计
* #### 极大似然估计法
先验概率的极大似然估计为： $ P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)}{N}, k=1,2,...,K $ 其中 I 是boolean函数

    设第 j 个特征 $ x^{(j)} $ 可能取值的集合为 {$ a_{j1}, a_{j2},...,a_{js_j} $}，则条件概率的极大似然估计为
    
     $$ P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^NI(y_i=c_k)} $$
    其中 j=1,2,...,n;  l=1,2,...,Sj;  k=1,2,...,K
    
* #### 贝叶斯估计法
用极大似然估计可能会出现某个所有估计的条件概率为0，这时会导致后延概率的计算结果为0，是分类不准确。为了解决这一问题，可以采用贝叶斯估计。

    给定 $ \lambda>0 $，则先验概率的贝叶斯估计为：
$$ P_{\lambda}(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)+\lambda}{N+K\lambda} $$

    条件概率的贝叶斯估计为：
    $$ P_{\lambda}(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i=c_k)+S_j\lambda} $$
    
    其中 Sj 为第j个特征的取值个数。可以看出当 $ \lambda=0 $ 时就变成了极大似然估计，特殊的当 $ \lambda=1 $,就是拉普拉斯平滑，这样就解决了在极大似然估计中概率为0的问题。 


***
## 导入相关包
这里导入 sklearn中的datasets包用于下载fetch_20newsgroups 数据作为本次模型的训练和测试数据


```python
import numpy as np
import collections
import re
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
```


```python
news = fetch_20newsgroups(data_home="./data/", subset='all')
```


```python
news.data[0]
```




    "From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>\nSubject: Pens fans reactions\nOrganization: Post Office, Carnegie Mellon, Pittsburgh, PA\nLines: 12\nNNTP-Posting-Host: po4.andrew.cmu.edu\n\n\n\nI am sure some bashers of Pens fans are pretty confused about the lack\nof any kind of posts about the recent Pens massacre of the Devils. Actually,\nI am  bit puzzled too and a bit relieved. However, I am going to put an end\nto non-PIttsburghers' relief with a bit of praise for the Pens. Man, they\nare killing those Devils worse than I thought. Jagr just showed you why\nhe is much better than his regular season stats. He is also a lot\nfo fun to watch in the playoffs. Bowman should let JAgr have a lot of\nfun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final\nregular season game.          PENS RULE!!!\n\n"



***
## 构建一个辅助函数用于处理文本中的符号，并切词


```python
# 处理文本中的符号
def process_text(data):
    processed_data = []
    for example in data:
        example = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，\n。？、~@#￥%……&*（）]+", " ", example)
        example = re.sub(r"\W", " ", example)
        processed_data.append(example.lower().split())
    return processed_data
```

***
## 划分数据集
使用 train_test_split()函数划分训练集和测试集

并对训练和测试数据集的文本进行预处理


```python
X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

((len(X_train), len(Y_train)), (len(X_test), len(Y_test)))
```




    ((14134, 14134), (4712, 4712))




```python
# 预处理文本
X_train = process_text(X_train)
X_test = process_text(X_test)
```

## 创建朴素贝叶斯模型类


```python
class NaiveBayesClassifier():
    def __init__(self, lambd):
        self.lambd = lambd
        self.vocabs = None
        self.prior_prob = None
        self.conditional_prob = None
    
    def fit(self, data_x, data_y):
        """训练，为了计算先验概率与条件概率"""
        start_time = time.time()
        print("开始构建词表...")
        self.vocabs = self.generate_vocab(data_x)
        print("生成训练数据矩阵")
        data_vec = self.convert_data_to_vec(data_x, self.vocabs)
        
        cate_num_k = len(set(data_y))
        
        # 计算先验概率
        print("开始计算先验概率...")
        self.prior_prob = {}
        for cate in set(data_y):
            # 采用贝叶斯估计
            self.prior_prob[cate] = (data_y.count(cate) + self.lambd) / (len(data_y) + cate_num_k * self.lambd)
        
        # 将数据按照类别不同划分为多个组
        group_data = {} 
        for cate in set(data_y):      
            sub_data_x = []
            for idx, example_label in enumerate(data_y):
                if example_label == cate:
                    sub_data_x.append(data_vec[idx])
            group_data[cate] = np.asarray(sub_data_x)
        
        # 计算每个类别的特征的条件概率
        print("开始计算条件概率...")
        # 所有类别下所有特征的不同取值的条件概率。
        self.conditional_prob = {}
        for cate in set(data_y):
            cate_data = group_data[cate]  
            # 某类子数据集的样本个数
            num_cate = cate_data.shape[0]
         
            feature_one_cond_prob = (np.sum(cate_data, axis=0) + self.lambd) / (num_cate + np.array([2] * cate_data.shape[1]) * self.lambd)
            feature_zero_cond_prob = (np.array([cate_data.shape[0]] * cate_data.shape[1]) - np.sum(cate_data, axis=0)) / (num_cate + np.array([2] * cate_data.shape[1]) * self.lambd)
                
            self.conditional_prob[cate] = [feature_zero_cond_prob, feature_one_cond_prob]
        stop_time = time.time()
        print("训练结束，耗时：{0} 秒".format(str(stop_time-start_time)))
        
        return self.prior_prob, self.conditional_prob
        
    
    def predict(self, data_test):
        """预测，在新的数据集上"""
        
        if self.prior_prob is None or self.conditional_prob is None or self.vocabs is None:
            raise NameError("模型未训练，没有可用的参数")
        test_data_vec = self.convert_data_to_vec(data_test, self.vocabs)
        
        # 测试集在每个类别上的后验概率
        test_cate_prob = np.zeros((test_data_vec.shape[0], len(self.prior_prob)))
        
        cate_idx = 0
        # 创建一个类别名称的列表，用于之后对结果的索引
        cates_name = []
        # 计算测试集每个样本在每个类别上的后验概率
        for cate in self.prior_prob.keys():
            cate_prior_prob = self.prior_prob[cate]
            cate_features_conditional_prob = self.conditional_prob[cate]
            
            cate_test_data_cond_prob = []
            
            for example in test_data_vec:
                example_feature_prob = 0.0
                for idx, feature_value in enumerate(example.tolist()):
                    example_feature_prob += cate_features_conditional_prob[int(feature_value)][idx]
                    
                cate_test_data_cond_prob.append(example_feature_prob)
            log_cate_union_prob = np.log(np.asarray(cate_test_data_cond_prob) + cate_prior_prob)
            test_cate_prob[:, cate_idx] = log_cate_union_prob
            cates_name.append(cate)        
            cate_idx += 1
            
        # 取后验概率最大的索引
        argmax_idx = np.argmax(test_cate_prob, axis=1)
        # 索引出类别名称
        test_cate_result = [cates_name[idx] for idx in argmax_idx]
        
        return test_cate_result
    
    def generate_vocab(self, data_x):
        """生成词表"""
        vocabs = set([])
        for example in data_x:
            vocabs = vocabs | set(example)
        return list(vocabs)
    
    def convert_data_to_vec(self, data_x, vocabs):
        """将文本数据集转换为向量，得到数据集矩阵，矩阵高为文本个数，宽为词汇表大小"""
        data_vec = np.zeros((len(data_x), len(vocabs)))
        for row, example in enumerate(data_x):
            for column, word in enumerate(example):
                if word in vocabs:
                    data_vec[row][column] = 1
        return data_vec
                    
```


```python
# 实例化一个 朴素贝叶斯分类器
NBClassifier = NaiveBayesClassifier(1.0)
```


```python
prior_prob, conditional_prob = NBClassifier.fit(X_train[0:2000], Y_train.tolist()[0:2000])
```

    开始构建词表...
    生成训练数据矩阵
    开始计算先验概率...
    开始计算条件概率...
    训练结束，耗时：435.9030749797821 秒
    
