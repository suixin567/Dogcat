import dataset
import tensorflow as tf
from datetime import timedelta
import numpy as np

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(10)#确定了种子
from tensorflow import set_random_seed
set_random_seed(20)


batch_size = 32

#Prepare input data
classes = ['dogs','cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2#验证集 占20%
img_size = 64#图像大小64*64
num_channels = 3
train_path='training_data'

#读数据 We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))




x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 1024#第一个全连接层的输出维度
num_iteration=8000

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="W")

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]), name="B")


#创建一个卷积层
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,
               name="conv"):  
    with tf.name_scope(name):
        ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
        weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        
        ## We create biases using the create_biases function. These are also trained.
        biases = create_biases(num_filters)
        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
        layer += biases
        layer = tf.nn.relu(layer)
        ## We shall be using max-pooling.  2*2
        layer = tf.nn.max_pool(value=layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
        #layer = tf.nn.relu(layer)
    
        return layer

    
#把卷积结果拉长，准备对接全连接层
def create_flatten_layer(layer,name="flatten"):
    with tf.name_scope(name):
        #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()
        ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()#计算特征维数。【？，8，8，64】 即：8*8*64
        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])
        return layer

#创建一个全连接层
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True,
             name="fc"):
    with tf.name_scope(name):
        #Let's define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)
        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.matmul(input, weights) + biases
        layer=tf.nn.dropout(layer,keep_prob=0.7)#防止过拟合，保留70%
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    
def model(learning_rate, use_two_conv, use_two_fc, hparam):
   
    session = tf.Session()
    #tf.reset_default_graph() 
    #卷积层
    layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

    layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

    layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 
    with tf.name_scope("cost"):
        y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
        y_pred_cls = tf.argmax(y_pred, dimension=1)#得到预测值
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)#对比是否正确
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#得到准确率
    session.run(tf.global_variables_initializer()) 
    #可视化 命令：tensorboard --logdir=./tensorboard/test1
    tenboard_dir = './tensorboard/test2/'
    writer = tf.summary.FileWriter(tenboard_dir + hparam)
    writer.add_graph(session.graph)
    
    #train
    saver = tf.train.Saver()
    for i in range(num_iteration):
            #取batch数据
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
    
            #用batch数据 填充 holder
            feed_dict_tr = {x: x_batch,y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch,y_true: y_valid_batch}
            #迭代一次（用train的数据）
            session.run(train_step, feed_dict=feed_dict_tr)
            if i % int(data.train.num_examples/batch_size) == 0: 
                val_loss = session.run(cost, feed_dict=feed_dict_val)#打印在验证集中的损失。
                #epoch = int(i / int(data.train.num_examples/batch_size))   
                epoch = data.train.epochs_done   
                #打印
                acc = session.run(accuracy, feed_dict=feed_dict_tr)
                val_acc = session.run(accuracy, feed_dict=feed_dict_val)
                msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
                print(msg.format(epoch + 1,i, acc, val_acc, val_loss))
                saver.save(session, './dogs-cats-model/dog-cat.ckpt',global_step=i) 
         

            
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)
    
    
def main():
  # You can try adding some more learning rates
    for learning_rate in [1E-4]:

    # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)
                # Actually run with the new settings
                model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()
