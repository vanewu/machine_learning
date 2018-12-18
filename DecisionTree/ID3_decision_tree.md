
## 决策树分类算法
决策树学习包括3个步骤：

* 特征选择
* 决策树的生成
* 决策树的修剪

决策树的特征选择通常采用特征增益或特征增益比进行选择

决策树的生成算法通常有Quinlan在1986年提出的ID3算法和1993年提出的C4.5算法

决策树的修剪往往通过极小化决策树的整体损失函数或代价函数进行，目的是为了减小决策树的过拟合问题，增大其泛化能力

***
## 信息熵
熵是表示随机变量不确定性的度量，设 X 是一个取有限个值的离散随机变量，其概率分布为：
$$ P(X = x_i) = p_i,   i=1,2,...,n $$

则随机变量X的熵为：
$$ H(X) = -\sum_{i=1}^{n}p_ilogp_i $$

熵越大，随机变量的不确定性就越大，且
$$ 0\leq H(p)\leq logn $$

***
## 条件熵
条件熵 & H(Y|X) & 表示在已知随机变量X的条件下随机变量Y的不确定性，定义为X给定的条件下Y的条件概率分布的熵对X的数学期望
$$ H(Y|X) = \sum_{i=1}^{n}p_iH(Y|X=x_i) $$

这里，& p_i = p(X=x_i), i=1,2,...,n &

***
## 信息增益
信息增益表示得知特征X的信息而使得Y的信息的不确定性减少的程度

特征A对训练数据集D的信息增益g(D,A),定义为集合D的经验熵H(D)与特征A给定的条件下D的敬仰条件熵H(D|A）之差，即：
$$ g(D,A)=H(D)-H(D|A) $$

一般的，熵H(Y)与条件熵H(Y|X)的差称为互信息，决策树学习中的信息增益等价于训练数据中类与特征的互信息

***
## 信息增益比
使用信息增益选择特征会使算法在选择特征时更倾向于选择特征取值个数多的特征，为了解决这一问题，可以采用信息赠增益比

特征A对训练数据D的信息增益比$ g_r(D,A) $为A对D的信息增益g(D,A)与训练数据集D关于特征A的值的熵$ H_A(D) $的比
$$ g_r(D,A)= \frac{g(D,A)}{H_A(D)} $$
$$ H_A(D) = -\sum_{i=1}^{n}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|} $$
n是特征A取值的个数



***
## 创建决策树模型类
该决策树生成算法使用 ID3 算法


```python
import numpy as np
```


```python
class ID3_Decision_Tree():
    def __init__(self):
        self.ID3_Tree = None
        
    def fit(self, data_set, features_name):
        """训练"""
        self.ID3_Tree = self.generate_tree(data_set, features_name)
        return self.ID3_Tree
    
    def predict(self, test_data, test_features_name):
        """预测"""
        model = self.ID3_Tree
        result = []
        for test_example in test_data:
            example_cate = self.classify(model, test_example, test_features_name)
            result.append(example_cate)
        return result
    
    def classify(self, ID3_Tree, test_example, test_features_name,):
        """辅助分类的函数"""
        if ID3_Tree is None:
            raise NameError("未经过训练，没有可用的树")
        first_key = list(ID3_Tree.keys())[0]
        second_dict = ID3_Tree[first_key]
        feature_index = test_features_name.index(first_key)
    
        for key in second_dict.keys():
            if test_example[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    example_category = self.classify(second_dict[key], test_example, test_features_name)
                else:
                    example_category = second_dict[key]
        return example_category
        
    def cal_shannon_entropy(self, data_set):
        """计算一个数据集的信息熵"""
        label_counts = {}
        data_len = len(data_set)

        for example in data_set:
            label = example[-1]
            if label in label_counts.keys():
                label_counts[label] += 1.0
            else:
                label_counts[label] = 1.0

        probs = np.asarray(list(label_counts.values())) / data_len
        shannon_entropy = - np.sum(probs * np.log2(probs))

        return shannon_entropy
    
    def split_data_set(self, data_set, axis):
        """根据给定的轴进行数据集划分"""
        column_single_value = set([example[axis] for example in data_set])
        all_sub_data = []
        for value in column_single_value:
            sub_data = []
            for example in data_set:
                sub_example = []
                if example[axis] == value:
                    sub_example.extend(example[:axis])
                    sub_example.extend(example[axis+1:])
                    sub_data.append(sub_example)
            all_sub_data.append(sub_data)
        return all_sub_data, column_single_value
    
    def choose_best_feature(self, data_set):
        """根据信息增益选择最好的特征进行数据集的划分"""
        # 减一是因为数据集最后一列为标签，不是特征
        features_len = len(data_set[0]) - 1
        data_len = len(data_set)
        data_set_entropy = self.cal_shannon_entropy(data_set)

        all_feautre_info_gain = np.zeros(features_len)

        for i in range(features_len):
            all_sub_data, _ = self.split_data_set(data_set, i)
            conditional_entropy = 0.0      
            for sub_data in all_sub_data:
                sub_prob = len(sub_data) / data_len
                sub_enropy = self.cal_shannon_entropy(sub_data)
                conditional_entropy += sub_prob * sub_enropy
            feature_info_gain = data_set_entropy - conditional_entropy
            all_feautre_info_gain[i] = feature_info_gain

        best_feature_idx = np.argmax(all_feautre_info_gain)

        return all_feautre_info_gain, best_feature_idx
    
    def vote(self, categories):
        """对多个类别进行投票多数表决"""
        cate_count = {}

        for cate in categories:
            if cate in cate_count.keys():
                cate_count[cate] += 1
            else:
                cate_count[cate] = 1

        voted_category = max(cate_count.items(), key=lambda x: x[1])[0]
        return voted_category

    def generate_tree(self, data_set, features_name):
        """生成决策树"""
        categories = [example[-1] for example in data_set]
        if categories.count(categories[0]) == len(categories):    # 都是同一个类别的时候停止划分
            return categories[0]
        if len(data_set[0]) == 1:    # 划分完了所有的特征，停止
            return self.vote(categories)

        _, best_feature_id = self.choose_best_feature(data_set)

        best_feature_name = features_name[best_feature_id]

        ID3_tree = {best_feature_name: {}}

        del features_name[best_feature_id]

        all_sub_data, all_unique_values = self.split_data_set(data_set, best_feature_id)

        for unique_value, sub_data in zip(all_unique_values, all_sub_data):
            sub_featrues_name = features_name[:]    # 这样赋值 sub_features_name 和 featrues_name指向的是不同对象
            ID3_tree[best_feature_name][unique_value] = self.generate_tree(sub_data, sub_featrues_name)

        return ID3_tree
```

## 简单测试


```python
# 实例化模型
id3_tree_object = ID3_Decision_Tree()
```


```python
# 创建训练集
data_set = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
feature_name = ['color', 'weight']
```


```python
# 训练生成决策树
id3_tree_object.fit(data_set, feature_name)
```




    {'color': {0: 'no', 1: {'weight': {0: 'no', 1: 'yes'}}}}




```python
# 创建测试集
test_data = [[0, 0],
             [0, 1],
             [1, 1]]
test_feature_name = ['color', 'weight']
```


```python
# 使用之前的决策树进行分类
id3_tree_object.predict(test_data, test_feature_name)
```




    ['no', 'no', 'yes']



## 模型的剪枝尚未编写，后面会继续更新
