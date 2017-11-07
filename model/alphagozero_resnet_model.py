# -*- coding: future_fstrings -*-
from model.resnet_model import *
from tensorflow.contrib.slim import prefetch_queue

class AlphaGoZeroResNet(ResNet):

    def __init__(self, hps, images, labels, zs, mode):
        self.zs = zs
        if hps is None:        
            hps = HParams(batch_size=1,
                           num_classes=362,
                           min_lrn_rate=0.0001,
                           lrn_rate=0.1,
                           num_residual_units=20,
                           use_bottleneck=False,
                           weight_decay_rate=0.0001,
                           relu_leakiness=0.1,
                           optimizer='mom')
        
        super().__init__(hps, images, labels, mode)
        
    # override _batch_norm to use tf.layers.batch_normalization
    def _batch_norm(self, name, x):
        """Build a batch norm layer for the model."""
        return tf.layers.batch_normalization(x,training=self.mode=='train',name=name,fused=True)

    # override _residual block to repliate AlphaGoZero architecture
    def _residual(self, x, in_filter, out_filter, stride):
        
        """Build a residual block for the model."""
        orig_x = x
        
        with tf.variable_scope('sub1'):
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
            # Batch normalisation
            x = self._batch_norm('bn1', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub2'):
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
            # Batch normalisation
            x = self._batch_norm('bn2', x)

        with tf.variable_scope('sub_add'):
            # A skip connection that adds the input to the block
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # override build graph
    def build_graph(self):
        
        """Build a whole graph for the model."""

        ''' https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

            build_graph
                 |---tower_loss
                      |---average_grads
                           |---add_L2_loss
                   grads---|
            tran_op---|

        '''
        
        with tf.device('/cpu:0'):
            
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.increase_global_step = self.global_step.assign_add(1)

            self.lrn_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, trainable=False)
            tf.summary.scalar('learning rate', self.lrn_rate)
            self.reinforce_dir = tf.Variable(1., dtype=tf.float32, trainable=False)
            
            if self.hps.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(1e-4)

            image_batches = tf.split(self.images,self.hps.num_gpu,axis=0)
            label_batches = tf.split(self.labels,self.hps.num_gpu,axis=0)
            z_batches = tf.split(self.zs,self.hps.num_gpu,axis=0)
            tower_grads = [None]*self.hps.num_gpu
            
        with tf.variable_scope(tf.get_variable_scope()):
            """Build the core model within the graph."""
            for i in range(self.hps.num_gpu):
                with tf.device(f'/gpu:{i}'):
                    with tf.name_scope(f'TOWER_{i}') as scope:
                        
                        image_batch, label_batch, z_batch = image_batches[i], label_batches[i], z_batches[i]
                        loss,move_acc,result_acc,temp = self._tower_loss(scope,image_batch,label_batch,z_batch,tower_idx=i)
                        tf.get_variable_scope().reuse_variables()
                        grad = self.optimizer.compute_gradients(loss)
                        if i == 0:
                            self.cost = loss
                            self.acc = move_acc
                            self.result_acc = result_acc
                            self.temp = temp
                        tower_grads[i] = grad

        grads = self._average_gradients(tower_grads)
        
        if self.mode == 'train':
            self._build_train_op(grads)
            
        self.summaries = tf.summary.merge_all()

    # overrride _build_model to be empty
    def _build_model(self):
        pass

    def _tower_loss(self,scope,image_batch,label_batch,z_batch,tower_idx):
        
        """Build the residual tower within the model."""
        with tf.variable_scope('init'):
                x = image_batch
                # A convolution of 256 filters of kernel size 3x3 with stride 1
                x = self._conv('init_conv', x, 3, 17, 256, self._stride_arr(1))
                # Batch normalisation
                x = self._batch_norm('initial_bn', x)
                # A rectifier non-linearity
                x = self._relu(x, self.hps.relu_leakiness)
                
        strides = [1, 1, 1]
        res_func = self._residual
        filters = [256, 256]
        
        with tf.variable_scope('res_block_0'):
            # _residual block to repliate AlphaGoZero architecture
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]))
            
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('res_block_%d' % i):
                # _residual block to repliate AlphaGoZero architecture
                x = res_func(x, filters[1], filters[1], self._stride_arr(1))
                
        with tf.variable_scope('policy_head'):
            # A convolution of 1 filter of kernel size 1x1 with stride 1
            logits = self._conv('policy_conv', x, 1, 256, 2, self._stride_arr(1))
            # Batch normalisation
            logits = self._batch_norm('policy_bn', logits)
            # A rectifier non-linearity
            logits = self._relu(logits, self.hps.relu_leakiness)
            # A fully connected linear layer that outputs a vector of
            # size 192^2 + 1 = 362 corresponding to logit probabilities
            # for all intersections and the pass move

            # defensive 1 step to temp annealling
            temp = tf.maximum(tf.train.exponential_decay(self.hps.temperature,self.global_step,1e4,0.8),1.)
            logits = tf.divide(self._fully_connected(logits, self.hps.num_classes,'policy_fc'),temp)
            prediction = tf.nn.softmax(logits)

        with tf.variable_scope('value_head'):
            # A convolution of 1 filter of kernel size 1x1 with stride 1
            value = self._conv('value_conv', x, 1, 256, 1, self._stride_arr(1))
            # Batch normalisation
            value = self._batch_norm('value_bn', value)
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a hidden layer of size 256
            value = self._fully_connected(value, 256, 'value_fc1')
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a scalar
            value = self._fully_connected(value, 1, 'value_fc2')
            # A tanh non-linearity outputting a scalar in the range [1, 1]
            value = tf.tanh(value)
            
        with tf.variable_scope('costs'):
            self.use_sparse_sotfmax = tf.constant(1, tf.int32, name="condition")
            def f1(): return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(label_batch,axis=1))
            def f2(): return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
            xent = tf.cond(self.use_sparse_sotfmax > 0, f1 , f2 )
            squared_diff = tf.squared_difference(z_batch,value)
            cost = tf.reduce_mean(xent, name='xent') + 0.01*tf.reduce_mean(squared_diff,name='squared_diff')
            cost += self._decay()
            tf.summary.scalar(f'cost_tower_{tower_idx}', cost)

        with tf.variable_scope('move_acc'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(label_batch,1))
            acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name=f'move_accu')
            tf.summary.scalar(f'move_accuracy_tower_{tower_idx}', acc)

        with tf.variable_scope('result_acc'):
            correct_prediction_2 = tf.equal(
                tf.sign(value), z_batch)
            result_acc = tf.reduce_mean(
                tf.cast(correct_prediction_2, tf.float32), name=f'result_accu')
            tf.summary.scalar(f'result_accuracy_tower_{tower_idx}', result_acc)

        return cost, acc, result_acc, temp

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
           Note that this function provides a synchronization point across all towers.
           Args:
              tower_grads: List of lists of (gradient, variable) tuples. The outer list
              is over individual gradients. The inner list is over the gradient
              calculation for each tower.
           Returns:
              List of pairs of (gradient, variable) where the gradient has been averaged
              across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
              # Add 0 dimension to the gradients to represent the tower.
              expanded_g = tf.expand_dims(g, 0)

              # Append on a 'tower' dimension which we will average over below.
              grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
            
        return average_grads

    # override build train op
    def _build_train_op(self, grads_vars):
        """Build training specific ops for the graph."""
        # defensive step 2 to clip norm
        clipped_grads,self.norm = tf.clip_by_global_norm([g for g,_ in grads_vars],self.hps.global_norm)

        # defensive step 3 check NaN
        # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
        grad_check = [tf.check_numerics(g,message='Nan Found!') for g in clipped_grads]
        with tf.control_dependencies(grad_check):
            """update_ops means to be batch_norm update"""
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_op = self.optimizer.apply_gradients(
                    zip(clipped_grads, [v for _,v in grads_vars]),
                    global_step=self.global_step, name='train_step')
                # since we include moving statistics in contrib.layer.batch_norm
                # there is no need to add _extra_train_ops
                #train_ops = [apply_op] + self._extra_train_ops
                #self.train_op = tf.group(*train_ops)
                self.train_op = apply_op

        
            
        
