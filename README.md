

Renilanmt
===
<br>


You have to download data and put the data in lanmt/mydata
--
<br>

Training:
--
```
python run.py --opt_dtok wmt14_ende --opt_batchtokens 4092 --opt_distill --opt_gen_pretrain --train
```

<br>

Tensorboard:
--
First, use the following script to connect server
```
ssh -L 16006:127.0.0.1:6006 ql@my_server_ip
```
Second, use the following script to open tensorboard
```
tensorboard --logdir log
```
Third, open the link in your local browser
```
http://127.0.0.1:16006
```
