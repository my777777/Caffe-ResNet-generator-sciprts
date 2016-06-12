1. 其中PlaiNet_CIFAR_10.py实现的是卷积神经网络的Caffe prototxt的Python脚本生成文件, 如果需要生成一个56的网络，根据公式, 层数=6*n+2, 运行方式如下：
   python PlaiNet_CIFAR_10 solver.prototxt train_val.prototxt --layer_number=9,9,9,0

2. 其中ResNet_CIFAR_10.py实现的是卷积神经网络的Caffe prototxt的Python脚本生成文件, 如果需要生成一个56的网络，根据公式, 层数=6*n+2, 运行方式如下：
   python ResNet_CIFAR_10 solver.prototxt train_val.prototxt --layer_number=9,9,9,0

3. 其中ResNet_ImageNet.py实现的是卷积神经网络的Caffe prototxt的Python脚本生成文件, 如果需要生成一个152的网络，根据公式, 层数=3*(n1 + n2 + n3 + n4) + 2, 运行方式如下：
   python ResNet_CIFAR_10 solver.prototxt train_val.prototxt --layer_number=3,8,36,3
