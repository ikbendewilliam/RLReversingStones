##Begin codeImport
import aigamingReversingStones
import tensorflow as tf
import numpy as np
import time
import datetime
import math
from random import randint
from random import choice
##End codeImport

##Begin codeInit
CONFIGURATIONS = [[8, 8], [14, 8], [4, 4]]
USE_CONFIGURATION = 0

BOARD_SIZE = CONFIGURATIONS[USE_CONFIGURATION]
print("Using configuration: \nBoard size: " + str(BOARD_SIZE[0]) + " by " + str(BOARD_SIZE[1]))
##End codeInit


##Begin codeInitialise
inputs_units = BOARD_SIZE[0] * BOARD_SIZE[1]
hidden_units = BOARD_SIZE[0] * BOARD_SIZE[1]
output_units = BOARD_SIZE[0] * BOARD_SIZE[1]

def initialise_tf():
    global input_positions, labels, learning_rate, W1, b1, h1, W2, b2, logits, probabilities, cross_entropy, train_step
    input_positions = tf.placeholder(tf.float32, shape=(1, inputs_units))
    labels          = tf.placeholder(tf.int64)
    learning_rate   = tf.placeholder(tf.float32, shape=[])

    # Generate hidden layer
    W1 = tf.Variable(tf.truncated_normal([inputs_units, hidden_units], stddev=0.1 / inputs_units**0.5))
    b1 = tf.Variable(tf.zeros([1, hidden_units]))
    h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)

    # Second ## -- linear classifier for action logits
    W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units], stddev=0.1 / hidden_units**0.5))
    b2 = tf.Variable(tf.zeros([1, output_units]))

    logits = tf.matmul(h1, W2) + b2
    probabilities = tf.nn.softmax(logits)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
##End codeInitialise

##Begin codeInitialiseGame
def initialise(player_index, player_ids, board_dimensions):
    return aigamingReversingStones.initialise(player_index=player_index, player_ids=player_ids, board_dimensions=board_dimensions)
##End codeInitialiseGame

##Begin codeMoveGame
def move(player_index, player_ids, gamestate, move):
    return aigamingReversingStones.move(player_index, player_ids, gamestate, move)
##End codeMoveGame


##Begin codeSave
def save_model(session, name='model.ckpt'):
    saver = tf.train.Saver()
    save_path = saver.save(session, "./models/" + name)
    print("Model saved in path: %s" % save_path)
##End codeSave

##Begin codeLoad
def load_model(name='model.ckpt', games = 100):
    global session
    tf.reset_default_graph()
    initialise_tf()
    
    session = tf.Session()
    tf.train.Saver().restore(session, "./models/" + name)
    benchmark(games)
##End codeLoad


##Begin codeBenchmark
def benchmark(games = 100):
    wins = draws = losses = 0
    for i in range(games):
        player_index = 0
        player_ids = ["p0", "p1"]
        result = initialise(player_index, player_ids, BOARD_SIZE)

        # Initialise logs for game
        board_log = []
        action_log = [[], []]
        while result["Result"] == "SUCCESS":
            player_index = result["GeneralGameStates"][-1]["Mover"]
            current_board = result["GeneralGameStates"][-1]["Board"]
            current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
            board_log.append([[i for i in j] for j in current_board])
            if player_index == 0:
                place_index, probs = guess_move(current_board, result["GeneralGameStates"][-1]["PossibleMoves"][player_index], False)
            else:
                possible_moves = [move[0] * BOARD_SIZE[1] + move[1] for move in result["GeneralGameStates"][-1]["PossibleMoves"][player_index]]
                place_index = choice(possible_moves)
            place = [int(place_index / BOARD_SIZE[1]), place_index % BOARD_SIZE[1]]
            place = {"Row": int(place[0]), "Column": int(place[1])}
            result = move(player_index, player_ids, result["GeneralGameStates"][-1], place)
            action_log[player_index].append(place_index)
        if result["WinnerIndex"] == 0:
            wins += 1
        elif result["WinnerIndex"] == -1:
            draws += 1
        else:
            losses += 1
        if (wins + draws + losses) % 100 == 0:
            print({"Wins":wins, "Draws":draws, "Losses":losses})
    t = wins + draws + losses
    return {"Wins":wins / t, "Draws":draws / t, "Losses":losses / t}
##End codeBenchmark

##Begin codeGuessMove
def guess_move(current_board, possible_moves, training):
    probs = session.run(probabilities, feed_dict={input_positions:[[cell for row in current_board for cell in row]]})[0]

    possible_moves = [move[0] * BOARD_SIZE[1] + move[1] for move in possible_moves]
    probs = [p if index in possible_moves else 0 for index, p in enumerate(probs)]
    probs = [round(p, 6) for p in probs]

    if sum(probs) > 0:
        probs = [p / sum(probs) for p in probs]
    else:
        probs = [1 / len(possible_moves) if index in possible_moves else 0 for index, p in enumerate(probs)]

    if training:
        place_index = np.random.choice(BOARD_SIZE[0] * BOARD_SIZE[1], p=probs)
    else:
        place_index = np.argmax(probs) # this is what you would do if you want to use the best option

    return place_index, probs[place_index]
##End codeGuessMove


##Begin codePlayGame
def play_game():
    player_index = 0
    player_ids = ["p0", "p1"]
    result = initialise(player_index, player_ids, BOARD_SIZE)
    
    # Initialise logs for game
    board_log = [[], []]
    action_log = [[], []]
    probs_log = [[], []]
    while result["Result"] == "SUCCESS":
        player_index = result["GeneralGameStates"][-1]["Mover"]
        current_board = result["GeneralGameStates"][-1]["Board"]
        current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
        board_log[player_index].append([[i for i in j] for j in current_board])
        place_index, prob = guess_move(current_board, result["GeneralGameStates"][-1]["PossibleMoves"][player_index], True)
        place = [int(place_index / BOARD_SIZE[1]), place_index % BOARD_SIZE[1]]
        place = {"Row": place[0], "Column": place[1]}
        result = move(player_index, player_ids, result["GeneralGameStates"][-1], place)
        probs_log[player_index].append(prob)
        action_log[player_index].append(place_index)
    return board_log, action_log, probs_log, result
##End codePlayGame

##Begin codePrep
# Reset training
initialise_tf()
init = tf.global_variables_initializer()

# Start TF session
session = tf.Session()
session.run(init)

all_boards = []
all_actions = []
all_probs = []
ALPHA = 0.00003     # step size
##End codePrep



##Begin codeTraining
NUMBER_OF_GAMES = 5000
begintime = time.time()
for game in range(NUMBER_OF_GAMES):
    board_log, action_log, probs_log, result = play_game()
    all_boards.append(board_log[-1][-1])
    all_actions.append(action_log)
    all_probs.append(probs_log)
    if (game + 1) % math.floor(NUMBER_OF_GAMES / 100) == 0 and game > 0:
        time_passed = math.floor(time.time() - begintime)
        print(str(game) + " / " + str(NUMBER_OF_GAMES) + \
                " (" + str(math.ceil(game / NUMBER_OF_GAMES * 100)) + "%) time passed: " + str(datetime.timedelta(seconds=time_passed)) + \
                " ETA: T-" + str(datetime.timedelta(seconds=math.floor(time_passed / game * NUMBER_OF_GAMES - time_passed))) + \
                " Avg of last set: " + str(sum([sum(x[0]) for x in all_probs[-math.floor(NUMBER_OF_GAMES / 100):]]) / math.floor(NUMBER_OF_GAMES / 100)))
    for player_index in range(2):
        ##Begin codeReward
        if result["WinnerIndex"] != player_index:
            rewards_log = [0 for i in probs_log[player_index]]
        else:
            rewards_log = [1 for i in probs_log[player_index]]
        ##End codeReward    
        for reward, current_board, action in zip(rewards_log, board_log[player_index], action_log[player_index]):
            session.run(train_step, feed_dict={input_positions:[[x for y in current_board for x in y]] , labels:[action], learning_rate:ALPHA * reward})

time_passed = math.floor(time.time() - begintime)
print("Total processed: " + str(len(all_actions)) + \
        " This took: " + str(datetime.timedelta(seconds=time_passed)) + \
        " Games / second: " + str(NUMBER_OF_GAMES / time_passed) + \
        " Total avg: " + str(sum([len(x) for x in all_actions]) / len(all_actions)))
save_model(session, name='model_' + str(BOARD_SIZE[0]) + 'x' + str(BOARD_SIZE[1]) + '.ckpt')
##End codeTraining

##Begin codeBenchmarkExecution
print(benchmark(1000))
##End codeBenchmarkExecution
