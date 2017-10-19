import sys
import board
import random
import numpy as np
import tensorflow as tf


def sigmoid(x):
    sig_out = (1 / (1 + np.exp(-x)))
    return sig_out

def state_matrix(state,height,width):
    M = np.zeros(shape=(height, width))
    i = height - 1
    j=0
    for column in state:
        for k in column:
            M[i, j] = k
            i -= 1
        i = height - 1
        j += 1
    return M

def insert_dict(diction,score,state):
    diction[state]=score

def play(human=False, n=1000, n_of_games=1):
    # Testing ConnectFour - mcts_uct()
    # height = 6
    height =3
    # width =7
    width = 4
    # target = 4
    target = 3
    initial = ((),) * width

    # game = board.ConnectFour(height=height, width=width, target=target)
    # state = initial
    # player = game.players[0]
    # computer = game.players[1]

    # here I try to add and define the dictionary that will hold everything together
    d_S_V1 = dict()
    d_S_V2 = dict()


    counter_for_games_played = 0 

    # before starting to play, we load the trained CNN network:

    # Parameters
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 128
    display_step = 10

    # Network Parameters
    n_input = 12  # board data input (img shape: 3*3)
    n_classes = 5  # total classes
    dropout = 0.75  # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 3, 4, 1])
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        # conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        # conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 3x4 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([3, 4, 1, 32]), ),
        # 3x4 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([3, 4, 32, 64])),
        # fully connected, 3*4*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([3 * 4 * 64, 1024])),
        # 1024 inputs, 5 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
    # up to here, we restored the trained CNN modek

        player1_wins=0
        player2_wins=0
        tie_games=0
        for i in xrange(1,n_of_games+1):    #we play n_of_games games
            game = board.ConnectFour(height=height, width=width, target=target)
            state = initial
            player = game.players[0]
            computer = game.players[1]

            while not game.terminal(state):


                if human:       #important: I COMMENTED EVERYTHING IN 'HUMAN' AND MADE 'HUMAN' A RANDOM PLAYER
                    prompt = 'Choose a move, choices are %s: ' % (game.actions(state),)
                    success = False
                    while not success:
                        choice = raw_input(prompt)
                        try:
                            action = int(choice)
                            state = game.result(state, action, player)
                            success = True
                        except ValueError:
                            pass
                        except Exception:
                            pass
                    # action=random.choice(game.actions(state))
                    # state=game.result(state,action,player)
                else:

                    input_data = np.zeros((1, 12))
                    input_data = input_data + game.state_in_a_row_for_first_NN(state)

                    action_vec = sess.run(pred, feed_dict={x: input_data, keep_prob: 1.})
                    CNN_action = np.argmax((action_vec))  #this is the action that the network wants
                    action = board.mcts_mAlphaGo(game, state, player, n, d_S_V2, CNN_action)
                    try:
                        #state = game.result(state, CNN_action, player) #this is for perfect player
                        state = game.result(state, action, player) #this is mAlphaGo
                    except:
                        #this is when the action value gives an illegal advice(shouldn't happen). when happens - use normal mcts
                        action1 = board.mcts_uct(game, state, player, n, d_S_V1)
                        state = game.result(state, action1, player)

                    # action = random.choice(game.actions(state))



                # Intermediate win check
                if game.terminal(state):
                    break

                # Computer plays now
                #****YOTAM: THIS WILL BE THE MCTS PLAYER
                #action_vec = sess.run(pred, feed_dict={x: game.state_in_a_row_for_first_NN(state), keep_prob: 1.})
                # input_data = np.zeros((1, 12))
                # input_data = input_data + game.state_in_a_row_for_first_NN(state)
                # # input_d = np.transpose(game.state_in_a_row_for_first_NN(state))
                # #print("the state input for CNN is: ", input_data)
                # action_vec = sess.run(pred, feed_dict={x: input_data, keep_prob: 1.})
                #
                # CNN_action = np.argmax(action_vec)
                # print CNN_action
                # #action = random.choice(game.actions(state))
                # state = game.result(state, CNN_action, computer)
                # #counter_for_games_played += 1  # YOTAM: added here
                action = board.mcts_uct(game, state, computer, n, d_S_V1)
                state = game.result(state, action, computer)

                #print 'Player 2 chose the best action to be %s' % action
            #print "counter for turns taken: ", counter_for_games_played  # YOTAM: added here
            #print game.pretty_state(state, False)
            #print game.better_pretty_state(state)  #FIXME: UNCOMMENT ME IF YOU WANT TO SEE THE STATES!
            #print
            outcome = game.outcome(state, player)

          #UNCOMMENT THIS SECTION IF YOU WANT TO SEE THE WINNER IN EVERY SINGLE GAME
            #******************
            if outcome == 1:
                print 'Player 1 wins game number ' + str(i) + '.'
                player1_wins+=1
            elif outcome == -1:
                print 'Player 2 wins game number ' + str(i) + '.'
                player2_wins+=1
            else:
                print 'Tie in game number ' + str(i) + '.'
                tie_games+=1
            #******************

            #enter the new database
            insert_dict(a,outcome,state)

        print 'player 1 - mAlphaGo - won ' + str(player1_wins) + ' games'
        print 'player 2  - MCTS - won ' + str(player2_wins) + ' games'
        print 'number of tie games: ' + str(tie_games)

n = 2000  # n is the budget
if len(sys.argv) > 1:
    try:
        n = int(sys.argv[1])
    except ValueError:
        pass

if '-n' in sys.argv:
    try:
        n = int(sys.argv[sys.argv.index('-n') + 1])
    except:
        pass

human = False  # mark as 'True' if you want to play against MCTS bot
if '-c' in sys.argv:
    human = False

print 'Number of Sample Iterations: ' + str(n)
print 'Human Player: ' + str(human)

a=dict()

play(human=human, n=n, n_of_games = 1)

