import datetime
import os

cmds = [
    "python ./CIFAR10_trainer.py --res_sub=transformation --model=ResNet --trigger_type=DA --adaptive=True --ablation_channel=-1 --ir=0.04 --repeats=3",
]

for cmd in cmds:
    try:
        print("Start running cmd: %s \nTime: %s \n" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("Finished cmd: %s \nTime: %s \n" % (cmd, datetime.datetime.now()))
    except:
        print("Failed running cmd: %s \nTime: %s \n" % (cmd, datetime.datetime.now()))
