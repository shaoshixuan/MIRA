这部分文件涉及到三部分数据：
1.  kg-item：用于item-id和asin做映射，获得asin字段
2. val-prediction-4000-for-reranking：我们前置训练pctr后，运行predict-pctr得到的预测结果
3. meta-数据集名称：亚马逊2023年商品元数据，present-asin和本次实验编码是对应的，需要映射后获取json文件中的title和description字段

** val-prediction-4000-for-reranking数据中的val不是采用验证集，实际我们就是采用rec-test-101数据进行的测试