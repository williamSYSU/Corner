1. save_model类型？
2. > no_cuda是否用到？
3. > dataprepare中的word2idx返回的变量bb是什么？
4. > datapreprare中的weakly_data()vocab没有赋值？
5. > dataprocess中的cal_sentence_index()是不是初始化句子向量的？


Datapro diff:
1. > 加载dataloader时，embedding没有转numpy
2. beginTrain_lstm()中的out1, out2, out3有view()操作，而Train.py中的weakly_train()没有

3. 改动1：调换位置：Datapro的line 294 和 line 295
4. 改动2：optimizer的参数设置
5. 改动3：在weakly_train()中，optimizer去掉zero_grad()

5. begintrain_lstm()和weakly_train()的有差别，将weakly_train()换成begintrain_lstm()后恢复正常。


Code Main issue:
1. 加载数据慢。主要在data_util.py中的DataPrepare.init()，因为要将词序列化，以及构建词典。
   - 解决办法：改用torchtext，但是要考虑句子的真实长度，可以创建一个Field，专门用于封装真实的句子长度。
   - 还要考虑如何采样的问题，从正例和负例中如何抽取
   
* 运行1：prepare_data + begintrain_lstm
x 运行2：prepare_data + begintrain_lstm + remove zero_grad()
x 运行3：push + begintrain_lstm + remove zero_grad()

2018-10-12 17:02
* 运行1：prepare_data + begintrain_lstm + to(device)
* 运行2：prepare_data + begintrain_lstm + cuda(device)
* 运行3：prepare_data + begintrain_lstm + cuda(device) + optimizer.zero_grad()
* 运行4：preapare_data + weakly_train + to(device) + view(batch_size,300)
* 运行5：preapare_data + weakly_train + to(device) + view(batch_size,300) + remove optimizer.zero_grad()

2018-10-12 18:35
运行1：preapare_data + weakly_train + to(device) + view(batch_size,300)

2018-10-12 19:04
+ 运行1：push + begintrain_lstm + to(device) + remove optimizer.zero_grad()
+ 运行2：push + begintrain_lstm + to(device) + add optimizer.zero_grad()
x 运行3：prepare_data + begintrain_lstm + to(device) + add optimizer.zero_grad()
x 运行4：prepare_data + begintrain_lstm + to(device) + add optimizer.zero_grad()
+ 运行5：push + weakly_train + to(device) + add optimizer.zero_grad()

2018-10-12 20:34:50
原weak_sr=0.8，weak_test_sample=1000
+ 运行7：push + weakly_train + to(device) + add optimizer.zero_grad() + weak_sr=0.7 + weak_test_sample=8000
x 运行7：prepare_data + weakly_train + to(device) + add optimizer.zero_grad() + weak_sr=0.7 + weak_test_sample=8000

2018-10-13 13:08:18
* 运行1：push + weakly_train + add zero_grad()
x 运行2：prepare_data + weakly_train + add zero_grad()
* 运行3：push + begintrain_lstm + add zero_grad()

2018-10-13 14:19:46
x 运行1：prepare_data + begintrain_lstm + add zero_grad()
* 运行2：push + weakly_train + add zero_grad()
* 运行3：prepare_data (func in) + weakly_train

2018-10-13 15:06:23
* 运行1：prepare_data (func in) + weakly_train + reset __init__()
x 运行2：prepare_data (func in) + weakly_train + reset __init__, sentence2vec and cal_sen_idx()