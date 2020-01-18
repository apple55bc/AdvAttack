# 预测运行环境：
 * RTX-2080TI
 * 数量：1
 * 深度学习框架：Tensorflow
 * requirements：
    * easydict==1.9
    * jieba==0.39
    * joblib==0.13.2
    * Keras==2.3.1
    * pinyin==0.4.0
    * tensorflow-gpu==1.14.0
    * pandas==0.25.0

# 预测运行说明
    ```bash
    cd code
    python3 aa_train_base_bert.py
    ```
这段代码训练阶段已经注释，会直接进行预测。   
预测结果保存在 /submit  文件夹下的 result.txt

# 代码结构说明
 * code  代码文件
    * bert4keras/  外部库代码
    * data/  数据处理代码
    * error_correct/  错字生成和纠错代码
    * aa_train_base_bert.py  训练和预测代码
 * data  官方数据、外部词数据和模型源文件
    * bert_roberta/  源模型文件
    * similar_words/  同义词、反义词数据
    * chars.dict  生成的train数据中出现的词（剔除单字的词）
    * law_word.txt  外部收集的法律相关词汇
    * stop_words.txt  外部停用词汇
    * test_set.csv  测试集
    * token_freq.txt  外部结巴词汇表
    * train_set.xml  训练集
 * model  存放训练的模型文件
    * bert_res/  训练完成的bert模型文件
    * detect/  训练完成的错词检测模型文件
    * error_maker_save/  纠错模型文件
    * tran_pre/  训练模型文件的词汇表部分
    * tran_pre_for_error_detect/  错词检测模型文件的词汇表部分
    
# 训练运行说明
准备Roberta-large模型源文件，解压放到 /data/bert_roberta 文件夹下
其他外部数据文件包括：   
 * data  官方数据、外部词数据和模型源文件
    * bert_roberta/  源模型文件
    * similar_words/  同义词、反义词数据
    * law_word.txt  外部收集的法律相关词汇
    * stop_words.txt  外部停用词汇
    * test_set.csv  测试集
    * token_freq.txt  外部结巴词汇表
    * train_set.xml  训练集
model文件下其它模型文件删除，否则某些模型会继续训练而不是重新开始
```bash
cd code
python3 aa_cfg.py
cd data
python3 aa_data_pre.py
cd ../error_correct
python3 ec_data_pre.py
python3 correct_by_statistics.py
```
此后，训练模型检测模型。（纠错模型在 correct_by_statistics.py 中已执行统计）   
train_error_detect.py 中的 line 75 ~ 76 互相注释，使用加载预训练模型
```bash
python3 train_error_detect.py
```
训练、预测6折交叉验证模型前，需要修改 aa_train_base_bert.py 代码。   
 * 1： line 30 ~ line 31  checkpoint_path 改为使用30行代码，31注释掉
 * 2： line 259 改为： res, eva_res = train(_fold, only_predict=False, need_val=False)
训练：
```bash
cd ..
python3 aa_train_base_bert.py
```