# MySK

## 0. 简介

**作者: David Ji**

**创建时间: July 7, 2016**

**功能:**

- 打包了[`scikit-learn`](http://scikit-learn.org/stable/index.html)中主要的分类器:

    - RandomForest
    - LogisticRegression
    - GBDT
    - SVC
    - LinearSVC
    - nuSVC

- 将数据集分为训练集和测试集输入后可以得到不同分类器及其性能比较.

## 1. 文件结构

```
.
│  .gitignore
│  LICENSE
│  README.md
│
├─code
│      Classifier.py
│      main.py
│      params.py
│
└─flowchart
        ml_map.png
```

- `README.md`: 本文件.

- `/code/Classifyer.py`: 类文件, 实现了上述分类器的打包.

- `/code/main.py`: 程序入口.

- `/code/params.py`: 参数文件, 可自定义数据库参数, 算法参数等.

- `/flowchart/ml_map.png`: 数据分析的流程图, 来自[`scikit-learn`](http://scikit-learn.org/stable/index.html), 仅供参考.

## 2. 使用说明

### 2.0 包依赖

- `MySQLdb`

- `numpy`

- `pandas`

- `sklearn`

### 2.1 params.py

- **DataBase Parameters**

    `USER`: 数据库连接用户名, `str`类型.
    
    `PASSWORD`: 数据库连接密码, `str`类型.
    
    `HOST`: 数据库连接主机, `str`类型.
    
    `PORT`: 数据库连接端口, `int`类型.
    
    `DB`: 目标数据库, `str`类型.
    
    `CHARSET`: 数据库字符编码格式, `str`类型.

- **Data Parameters**

    `DB_SQL`: 读取数据的SQL语句, `str`类型.

    `DB_TABLE`: 要读取的数据表, `str`类型.

    `FIELDS`: 要读取的字段列表, `list`类型, 空list表示读取所有字段.

    `CONDITIONS`: 读取时的限制条件, `list`类型, 空list表示没有`WHERE`条件.

    `LIMIT`: 要读取的数据量, `str`类型.

    > 注:
    >
    > 1. 如果`SELECT [FIELDS] FROM [TABLE] WHERE [COND1 AND ...] LIMIT [NUM]`形式可以满足查询需求, 则设置`DB_TABLE`, `FIELDS`, `CONDITIONS`和`LIMIT`参数即可, `DB_SQL`参数无需设置;
    >
    > 2. 若上述查询表达形式不能满足需求, 则需设置`DB_SQL`为完整的查询语句, 其它参数无需设置.
    > 3. 若`CONDITIONS`参数中含有多个**条件表达式**, 会认为这些表达式之间是`AND`关系, 若需要指定`OR`关系的条件, 可以将其合并为一个条件写入`CONDITIONS`中.

- **Algorithm Parameters**

    `FEATURE_STATR`: 特征字段起始下标(含), `int`类型.

    `FEATURE_END`: 特征字段结束下标(不含), `int`类型.

    `LABEL`: 标签字段下标, `int`类型.

    `TRAIN_SIZE`: 训练集百分比, `float`类型.

    > 注:
    >
    > 这种指定方式基于数据集中的特征字段是连续排列的, 如果源数据中的特征字段不是连续排列的, 可以通过`FIELDS`参数来调整顺序, 使其连续.

### 2.2 Classifier.py
    
1. 初始化Classifier

    ```
    clfs = Classifier(X_train, X_test, y_train, y_test)
    ```

2. Classifier对象具有的属性

    - **`X_train`**: 训练集中的特征集

    - **`X_test`**: 训练集中的标签集

    - **`y_train`**: 测试集中的特征集

    - **`y_test`**: 测试集中的标签集

    - **`RF`**: RandomForest分类器

    - **`LR`**: LogisticRegression分类器

    - **`GBDT`**: GBDT分类器

    - **`SVM_SVC`**: SVC分类器

    - **`SVM_LinearSVC`**: LinearSVC分类器

    - **`SVM_nuSVC`**: nuSVC分类器

    - **`comparison`**: 各分类器的性能比较

3. Classifier对象具有的方法

    - **`predit(clf, proba=False)`**: 使用`clf`分类器预测`X_test`数据集, `proba=False`时返回预测结果, `proba=True`时返回各类别的预测概率.

    - **`report(preds)`**: 根据预测结果`preds`得出此次分类预测的报告.

    - **`matrix(preds)`**: 根据预测结果`preds`得出此次分类预测的混淆矩阵.

    - **`performance(preds)`**: 根据预测结果`preds`得出此次分类的性能.
        
        - `accuracy_score`: [正确率](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), 预测对的样本数 / 总样本数

        - `precision_score`: [准确率](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html), 各个类别的准确率的加权均值, 某类别的准确率为: 真正数 / (真正数 + 假正数)

        - `recall_score`: [召回率](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html),  各个类别的召回率的加权均值, 某类别的召回率为: 真正数 / (真正数 + 假负数)

        - `f1_score`: [f1值](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), 各个类别的f1值的加权均值, 某类别的f1值为: 2 * (准确率 * 召回率) / (准确率 + 召回率)

        - `jaccard_similarity_score`: [杰卡德相似系数](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html), [计算方法](https://en.wikipedia.org/wiki/Jaccard_index)

        - `hamming_loss`: [汉明损失](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html), 该指标衡量了预测所得标签与样本实际标签之间的不一致程度, 即样本具有标签`yi`但未被识别, 或不具备标签`yi`却被误判的可能性. [计算方法](https://www.kaggle.com/wiki/HammingLoss)

        - `zero_one_loss`: [0-1损失](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html), [计算方法](https://en.wikipedia.org/wiki/Loss_function)


4. 其它

    1. 各分类器训练时使用的参数都是[`scikit-learn`](http://scikit-learn.org/stable/index.html)中的默认参数, 若需修改参数, 可更改`Classifier.py`中相应分类器的初始化语句.

    2. 通过`Classifier`对象得到的分类器具有[`scikit-learn`](http://scikit-learn.org/stable/index.html)中相应分类器拥有的所有属性和方法. 具体可参考:

        > - [`RandomForest`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

        > - [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

        > - [`GBDT`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

        > - [`SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

        > - [`LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

        > - [`nuSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)

### 2.3 main.py
`main()`函数中已经得到了分类器集合`clfs`, 更多需求可在后面添加.


---

> **参考资料**:
>
1. scikit-learn, http://scikit-learn.org/stable/index.html
2. Wikipedia, https://en.wikipedia.org/wiki/Main_Page
3. Kaggle, https://www.kaggle.com/


