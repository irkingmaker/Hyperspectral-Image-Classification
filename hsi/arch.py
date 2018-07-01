import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import s_preproc as s_pp

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

tf.reset_default_graph
X=tf.placeholder(tf.float32,shape=[None,3,3,224])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
# define model
def complex_model(X,y,is_training):
    #3x3 convlayer
    Wconv1 = tf.get_variable("Wconv1", shape=[1, 1, 224, 500])
    bconv1 = tf.get_variable("bconv1", shape=[500])

    a1=tf.nn.conv2d(X,Wconv1,strides=[1,1,1,1],padding='VALID')+bconv1
    
    a1_relu=tf.nn.relu(a1)
    
    #a1_bn=tf.layers.batch_normalization(inputs=a1_relu,axis=-1,center=True,scale=True,training = is_training)
    
    #a1_pool = tf.layers.max_pooling2d(inputs=a1_bn, pool_size=[2, 2], strides=2)
    
    Wconv2 = tf.get_variable("Wconv2", shape=[1, 1, 500, 100])
    bconv2 = tf.get_variable("bconv2", shape=[100])

    a1_conv2=tf.nn.conv2d(a1_relu,Wconv2,strides=[1,1,1,1],padding='VALID')+bconv2

    a1_relu2=tf.nn.relu(a1_conv2)
    
    a1_flat=tf.layers.flatten(a1_relu2)
    
    a1_aff=tf.layers.dense(a1_flat,1344,activation=tf.nn.relu)
    
    a1_aff2=tf.layers.dense(a1_flat,16,activation=None)
    
    return a1_aff2

y_out=complex_model(X,y,is_training)

mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,16), logits=y_out))
optimizer = tf.train.RMSPropOptimizer(1e-3)

X_train=s_pp.x_train
y_train=s_pp.x_train_labels
print('h')

with tf.Session() as sess:
    with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)