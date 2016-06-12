#!/usr/bin/env python
"""
Generate the residule learning network.
Author: Yemin Shi
Email: shiyemin@pku.edu.cn

MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('solver_file',
                        help='Output solver.prototxt file')
    parser.add_argument('train_val_file',
                        help='Output train_val.prototxt file')
    parser.add_argument('--layer_number', nargs='*',
                        help=('Layer number for each layer stage.'),
                        default=['3,8,36,3'])
    parser.add_argument('-t', '--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer(n):
    data_layer_str = '''name: "ResNet-%d"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "examples/res_net/ilsvrc12_train_lmdb"
    batch_size: 16
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "examples/res_net/ilsvrc12_val_lmdb"
    batch_size: 8
    backend: LMDB
  }
}\n''' % (3 * n + 2)
    return data_layer_str

def generate_conv1_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
     lr_mult: 2
     decay_mult: 0
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
      std: 0.010
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}\n'''%(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
      std: 0.010
    }
    bias_term: false
  }
}\n'''%(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str



def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}\n'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}\n'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}\n'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}\n'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}\n
layer {
  name: "acc/top-1"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-1"
  include {
    phase: TEST
  }
}\n
layer {
  name: "acc/top-5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}\n'''%(bottom, bottom, bottom)
    return softmax_loss_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  batch_norm_param {
      use_global_stats: false
  }
}\n'''%(layer_name, bottom, top)
    return bn_layer_str

def generate_scale_layer(layer_name, bottom, top):
    scale_layer_str = '''layer {
  name: "%s"  
  type: "Scale"
  bottom: "%s"
  top: "%s"
  scale_param {
	bias_term: true
  }
  param { 
        lr_mult: 1
  }
}\n ''' % (layer_name, bottom, top)
    return scale_layer_str

def generate_state_layer_names(stage, seq):
    '''
    layer_name: res
    stage: 1
    branch: 2
    seq: 1
    
    examples: res_1_branch2_1
              scale_2_branch2_2

              res_stage_1_seq_1
    '''
    layer_names = []
    
    for i in xrange(3):
        sub_branches = []
        sub_branches.append('res_stage_' + str(stage)    + '_' + str(seq) + '_' + str(i + 1))
        sub_branches.append('bn_stage_' + str(stage)     + '_' + str(seq) + '_' + str(i + 1))
        sub_branches.append('scale_stage_' + str(stage)  + '_' + str(seq) + '_' + str(i + 1))
        sub_branches.append('res_stage_' + str(stage)    + '_' + str(seq) + '_' + str(i + 1) + '_relu')
        
        layer_names.append(sub_branches)
    
    layer_names.append('res_%d_%d' % (stage, seq))    
    
    return layer_names

def generate_train_val():
    args = parse_args()
    stage_n = args.layer_number[0]
    stage_n = [int(n) for n in args.layer_number[0].split(',')]

    network_str = generate_data_layer(sum(stage_n))
    '''before stage'''
    last_top = 'data'
    network_str += generate_conv1_layer(7, 64, 2, 3, 'conv1', last_top, 'conv1', 'msra')
    network_str += generate_bn_layer('bn_conv1', 'conv1', 'conv1')
    network_str += generate_scale_layer('scale_conv1', 'conv1', 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1', 'pool1')
    
    '''stage 1'''
    stage = 1
    last_top = 'pool1'    
    
    # branch1
    # def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    network_str += generate_conv_layer(1, 256, 1, 0, 'res_1_branch1', last_top, 'res_1_branch1', 'msra')
    network_str += generate_bn_layer('bn_1_branch1', 'res_1_branch1', 'res_1_branch1')
    network_str += generate_scale_layer('scale_1_branch1', 'res_1_branch1', 'res_1_branch1')
    last_output = 'res_1_branch1'

    # branch2
    print stage_n
    for l in xrange(1, stage_n[0] + 1):
        branch2_layer_names = generate_state_layer_names(stage, l)
        print(branch2_layer_names)        
        
        sub_branch1 = branch2_layer_names[0]
        network_str += generate_conv_layer(1, 64, 1, 0, sub_branch1[0], last_top, sub_branch1[0], 'msra')
        network_str += generate_bn_layer(sub_branch1[1], sub_branch1[0], sub_branch1[0])
        network_str += generate_scale_layer(sub_branch1[2], sub_branch1[0], sub_branch1[0])
        network_str += generate_activation_layer(sub_branch1[3], sub_branch1[0], sub_branch1[0], 'ReLU')
        
        sub_branch2 = branch2_layer_names[1]
        network_str += generate_conv_layer(3, 64, 1, 1, sub_branch2[0], sub_branch1[0], sub_branch2[0], 'msra')
        network_str += generate_bn_layer(sub_branch2[1], sub_branch2[0], sub_branch2[0])
        network_str += generate_scale_layer(sub_branch2[2], sub_branch2[0], sub_branch2[0])
        network_str += generate_activation_layer(sub_branch2[3], sub_branch2[0], sub_branch2[0], 'ReLU')
        
        sub_branch3 = branch2_layer_names[2]
        network_str += generate_conv_layer(1, 256, 1, 0, sub_branch3[0], sub_branch2[0], sub_branch3[0], 'msra')
        network_str += generate_bn_layer(sub_branch3[1], sub_branch3[0], sub_branch3[0])        
        network_str += generate_scale_layer(sub_branch3[2], sub_branch3[0], sub_branch3[0])

        # merge(eltwise)
        elt_wise_layer_name = branch2_layer_names[3]
        network_str += generate_eltwise_layer(elt_wise_layer_name, last_output, sub_branch3[0], elt_wise_layer_name, 'SUM')
        network_str += generate_activation_layer(elt_wise_layer_name + '_relu', elt_wise_layer_name, elt_wise_layer_name, 'ReLU')
        
        last_top = elt_wise_layer_name
        last_output = elt_wise_layer_name

    '''stage 2'''
    stage = 2
    
    # branch1
    network_str += generate_conv_layer(1, 512, 2, 0, 'res_2_branch1', last_top, 'res_2_branch1', 'msra')
    network_str += generate_bn_layer('bn_2_branch1', 'res_2_branch1', 'res_2_branch1')
    network_str += generate_scale_layer('scale_2_branch1', 'res_2_branch1', 'res_2_branch1')
    last_output = 'res_2_branch1'
    
    # branch2
    for l in xrange(1, stage_n[1] + 1):
        branch2_layer_names = generate_state_layer_names(stage, l)         
        
        if l == 1:
            stride = 2
        else:
            stride = 1            
        
        sub_branch1 = branch2_layer_names[0]
        network_str += generate_conv_layer(1, 128, stride, 0, sub_branch1[0], last_top, sub_branch1[0], 'msra')
        network_str += generate_bn_layer(sub_branch1[1], sub_branch1[0], sub_branch1[0])
        network_str += generate_scale_layer(sub_branch1[2], sub_branch1[0], sub_branch1[0])
        network_str += generate_activation_layer(sub_branch1[3], sub_branch1[0], sub_branch1[0], 'ReLU')
        
        sub_branch2 = branch2_layer_names[1]
        network_str += generate_conv_layer(3, 128, 1, 1, sub_branch2[0], sub_branch1[0], sub_branch2[0], 'msra')
        network_str += generate_bn_layer(sub_branch2[1], sub_branch2[0], sub_branch2[0])
        network_str += generate_scale_layer(sub_branch2[2], sub_branch2[0], sub_branch2[0])
        network_str += generate_activation_layer(sub_branch2[3], sub_branch2[0], sub_branch2[0], 'ReLU')
        
        sub_branch3 = branch2_layer_names[2]
        network_str += generate_conv_layer(1, 512, 1, 0, sub_branch3[0], sub_branch2[0], sub_branch3[0], 'msra')
        network_str += generate_bn_layer(sub_branch3[1], sub_branch3[0], sub_branch3[0])        
        network_str += generate_scale_layer(sub_branch3[2], sub_branch3[0], sub_branch3[0])        
        
        # merge(eltwise)
        elt_wise_layer_name = branch2_layer_names[3]
        network_str += generate_eltwise_layer(elt_wise_layer_name, last_output, sub_branch3[0], elt_wise_layer_name, 'SUM')
        network_str += generate_activation_layer(elt_wise_layer_name + '_relu', elt_wise_layer_name, elt_wise_layer_name , 'ReLU')

        last_top = elt_wise_layer_name
        last_output = elt_wise_layer_name   
    
    '''stage 3'''
    stage = 3
    
    # branch1
    network_str += generate_conv_layer(1, 1024, 2, 0, 'res_3_branch1', last_top, 'res_3_branch1', 'msra')
    network_str += generate_bn_layer('bn_3_branch1', 'res_3_branch1', 'res_3_branch1')
    network_str += generate_scale_layer('scale_3_branch1', 'res_3_branch1', 'res_3_branch1')
    last_output = 'res_3_branch1'
    
    # branch2
    for l in xrange(1, stage_n[2] + 1):
        branch2_layer_names = generate_state_layer_names(stage, l)
        if l == 1:
            stride = 2
        else:
            stride = 1
        
        sub_branch1 = branch2_layer_names[0]
        network_str += generate_conv_layer(1, 256, stride, 0, sub_branch1[0], last_top, sub_branch1[0], 'msra')
        network_str += generate_bn_layer(sub_branch1[1], sub_branch1[0], sub_branch1[0])
        network_str += generate_scale_layer(sub_branch1[2], sub_branch1[0], sub_branch1[0])
        network_str += generate_activation_layer(sub_branch1[3], sub_branch1[0], sub_branch1[0], 'ReLU')
        
        sub_branch2 = branch2_layer_names[1]
        network_str += generate_conv_layer(3, 256, 1, 1, sub_branch2[0], sub_branch1[0], sub_branch2[0], 'msra')
        network_str += generate_bn_layer(sub_branch2[1], sub_branch2[0], sub_branch2[0])
        network_str += generate_scale_layer(sub_branch2[2], sub_branch2[0], sub_branch2[0])
        network_str += generate_activation_layer(sub_branch2[3], sub_branch2[0], sub_branch2[0], 'ReLU')
        
        sub_branch3 = branch2_layer_names[2]
        network_str += generate_conv_layer(1, 1024, 1, 0, sub_branch3[0], sub_branch2[0], sub_branch3[0], 'msra')
        network_str += generate_bn_layer(sub_branch3[1], sub_branch3[0], sub_branch3[0])
        network_str += generate_scale_layer(sub_branch3[2], sub_branch3[0], sub_branch3[0])
        
        # merge(eltwise)
        elt_wise_layer_name = branch2_layer_names[3]
        network_str += generate_eltwise_layer(elt_wise_layer_name, last_output, sub_branch3[0], elt_wise_layer_name, 'SUM')
        network_str += generate_activation_layer(elt_wise_layer_name + '_relu', elt_wise_layer_name, elt_wise_layer_name, 'ReLU')

        last_top = elt_wise_layer_name
        last_output = elt_wise_layer_name       
    
    '''stage 4'''
    stage = 4
    
    # branch1
    network_str += generate_conv_layer(1, 2048, 2, 0, 'res_4_branch1', last_top, 'res_4_branch1', 'msra')
    network_str += generate_bn_layer('bn_4_branch1', 'res_4_branch1', 'res_4_branch1')
    network_str += generate_scale_layer('scale_4_branch1', 'res_4_branch1', 'res_4_branch1')
    last_output = 'res_4_branch1'    
    
    # branch2
    for l in xrange(1, stage_n[3] + 1):
        branch2_layer_names = generate_state_layer_names(stage, l)        
        if l == 1:
            stride = 2
        else:
            stride = 1
        
        sub_branch1 = branch2_layer_names[0]
        network_str += generate_conv_layer(1, 512, stride, 0, sub_branch1[0], last_top, sub_branch1[0], 'msra')
        network_str += generate_bn_layer(sub_branch1[1], sub_branch1[0], sub_branch1[0])
        network_str += generate_scale_layer(sub_branch1[2], sub_branch1[0], sub_branch1[0])
        network_str += generate_activation_layer(sub_branch1[3], sub_branch1[0], sub_branch1[0], 'ReLU')
        
        sub_branch2 = branch2_layer_names[1]
        network_str += generate_conv_layer(3, 512, 1, 1, sub_branch2[0], sub_branch1[0], sub_branch2[0], 'msra')
        network_str += generate_bn_layer(sub_branch2[1], sub_branch2[0], sub_branch2[0])
        network_str += generate_scale_layer(sub_branch2[2], sub_branch2[0], sub_branch2[0])
        network_str += generate_activation_layer(sub_branch2[3], sub_branch2[0], sub_branch2[0], 'ReLU')
        
        sub_branch3 = branch2_layer_names[2]
        network_str += generate_conv_layer(1, 2048, 1, 0, sub_branch3[0], sub_branch2[0], sub_branch3[0], 'msra')
        network_str += generate_bn_layer(sub_branch3[1], sub_branch3[0], sub_branch3[0])
        network_str += generate_scale_layer(sub_branch3[2], sub_branch3[0], sub_branch3[0])
        
        # merge(eltwise)
        elt_wise_layer_name = branch2_layer_names[3]
        network_str += generate_eltwise_layer(elt_wise_layer_name, last_output, sub_branch3[0], elt_wise_layer_name, 'SUM')
        network_str += generate_activation_layer(elt_wise_layer_name + '_relu', elt_wise_layer_name, elt_wise_layer_name , 'ReLU')

        last_top = elt_wise_layer_name
        last_output = elt_wise_layer_name        
    
    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool5', last_top, 'pool5')
    network_str += generate_fc_layer(1000, 'fc1000', 'pool5', 'fc1000', 'msra')
    network_str += generate_softmax_loss('fc1000')
    return network_str

def generate_solver(train_val_name):
    solver_str = '''net: "%s"
test_iter: 1000
test_interval: 2000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
gamma: 0.1
max_iter: 600000
stepvalue: 320000
stepvalue: 480000
momentum: 0.9
weight_decay: 0.0001
snapshot: 10000
snapshot_prefix: "res_net"
solver_mode: GPU'''%(train_val_name)
    return solver_str

def main():
    args = parse_args()
    solver_str = generate_solver(args.train_val_file)
    network_str = generate_train_val()
    fp = open(args.solver_file, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(args.train_val_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
