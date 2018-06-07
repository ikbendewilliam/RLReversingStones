import aigamingReversingStones
import tensorflow as tf
import numpy as np
import time
import datetime
import math
import time
from random import randint
from random import choice

CONFIGURATIONS = [[8, 8], [14, 8], [4, 4]]
USE_CONFIGURATION = 0

BOARD_SIZE = CONFIGURATIONS[USE_CONFIGURATION]
DIVIDER = max(BOARD_SIZE[0],BOARD_SIZE[1])  * 4
print("Using configuration: \nBoard size: " + str(BOARD_SIZE[0]) + " by " + str(BOARD_SIZE[1]))

inputs_units = BOARD_SIZE[0] * BOARD_SIZE[1]
hidden_units = BOARD_SIZE[0] * BOARD_SIZE[1]
output_units = BOARD_SIZE[0] * BOARD_SIZE[1]


def initialise(player_index, player_ids, board_dimensions):
    return aigamingReversingStones.initialise(player_index=player_index, player_ids=player_ids, board_dimensions=board_dimensions)

def move(player_index, player_ids, gamestate, move):
    return aigamingReversingStones.move(player_index, player_ids, gamestate, move)


def initialize_tf():
    global input_positions, labels, learning_rate, W1, b1, h1, W3, b3, logits, probabilities, cross_entropy, train_step
    input_positions = tf.placeholder(tf.float32, shape=(1, inputs_units))# + 1
    labels =          tf.placeholder(tf.int64)
    learning_rate =   tf.placeholder(tf.float32, shape=[])

    # Generate hidden layer
    W1 = tf.Variable(tf.truncated_normal([inputs_units, hidden_units],#**2 + 1
                 stddev=0.1 / np.sqrt(float(inputs_units))))#**2
    b1 = tf.Variable(tf.zeros([1, hidden_units]))
    h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)

    # Second layer -- linear classifier for action logits
    W3 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
                 stddev=0.1 / np.sqrt(float(hidden_units))))
    b3 = tf.Variable(tf.zeros([1, output_units]))

    logits = tf.matmul(h1, W3) + b3
    probabilities = tf.nn.softmax(logits)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


def guess_move(current_board, possible_moves, training):
    probs = sess.run(probabilities, feed_dict={input_positions:[[cell for row in current_board for cell in row]]})[0]

    possible_moves = [move[0] * BOARD_SIZE[1] + move[1] for move in possible_moves]
    probs = [p * (index in possible_moves) for index, p in enumerate(probs)]
    probs = [int(p * 1000 * 1000) / 1000 / 1000 for p in probs]

    if sum(probs) > 0:
        probs = [p / sum(probs) for p in probs]
    else:
        probs = [1 / len(probs) for p in probs]
    counter = 0
    while True:
        if counter >= 10:
            place_index = np.random.choice(BOARD_SIZE[0] * BOARD_SIZE[1])
        elif training == True:
            place_index = np.random.choice(BOARD_SIZE[0] * BOARD_SIZE[1], p=probs)
        else:
            place_index = np.argmax(probs)
        if place_index in possible_moves:
            break
        counter += 1
    return place_index, probs[place_index]


TRAINING = True
def play_game(sess, training=TRAINING):
    player_index = 0
    player_ids = ["p0", "p1"]
    result = initialise(player_index, player_ids, BOARD_SIZE)
    
    # Initialize logs for game
    board_position_log = []
    action_log = [[], []]
    gains_log = [[], []]
    probs_log = [[], []]
    while result["Result"] == "SUCCESS":
        player_index = result["GeneralGameStates"][-1]["Mover"]
        current_board = result["GeneralGameStates"][-1]["Board"]
        current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
        board_position_log.append([[i for i in j] for j in current_board])
        place_index, prob = guess_move(current_board, result["GeneralGameStates"][-1]["PossibleMoves"][player_index], training)
        place = [int(place_index / BOARD_SIZE[1]), place_index % BOARD_SIZE[1]]
        place = {"Row": place[0], "Column": place[1]}
        result = move(player_index, player_ids, result["GeneralGameStates"][-1], place)
        current_board = result["GeneralGameStates"][-1]["Board"]
        current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
        gains = get_gains(board_position_log[-1], current_board, 0)
        probs_log[player_index].append(prob)
        action_log[player_index].append(place_index)
        gains_log[player_index].append(gains)
    return board_position_log, action_log, gains_log, probs_log, result


def get_gains(board_last, board_new, role):
    board_last_count = sum([1 if cell == role else 0 for row in board_last for cell in row])
    board_new_count = sum([1 if cell == role else 0 for row in board_new for cell in row])
    return (board_new_count - board_last_count) / board_new_count


def benchmark(games = 100):
    wins = draws = losses = 0
    for i in range(games):
        player_index = 0
        player_ids = ["p0", "p1"]
        result = initialise(player_index, player_ids, BOARD_SIZE)

        # Initialize logs for game
        board_position_log = []
        action_log = [[], []]
        gains_log = [[], []]
        while result["Result"] == "SUCCESS":
            player_index = result["GeneralGameStates"][-1]["Mover"]
            current_board = result["GeneralGameStates"][-1]["Board"]
            current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
            board_position_log.append([[i for i in j] for j in current_board])
            if player_index == 0:
                place_index, probs = guess_move(current_board, result["GeneralGameStates"][-1]["PossibleMoves"][player_index], False)
            else:
                possible_moves = [move[0] * BOARD_SIZE[1] + move[1] for move in result["GeneralGameStates"][-1]["PossibleMoves"][player_index]]
                place_index = choice(possible_moves)
            place = [int(place_index / BOARD_SIZE[1]), place_index % BOARD_SIZE[1]]
            place = {"Row": int(place[0]), "Column": int(place[1])}
            result = move(player_index, player_ids, result["GeneralGameStates"][-1], place)
            current_board = result["GeneralGameStates"][-1]["Board"]
            current_board = [[-1 if cell == -1 else 0 if cell == result["GeneralGameStates"][-1]["Roles"][player_index] else 1 for cell in row] for row in current_board]
            gains = get_gains(board_position_log[-1], current_board, 0)
            action_log[player_index].append(place_index)
            gains_log[player_index].append(gains)
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

def save_model(session, name='model.ckpt'):
    saver = tf.train.Saver()
    save_path = saver.save(session, "./models/" + name)
    print("Model saved in path: %s" % save_path)


# Reset training
initialize_tf()
init = tf.global_variables_initializer()

# Start TF session
sess = tf.Session()
sess.run(init)

all_boards = []
all_actions = []
all_gains = []
all_probs = []
TRAINING = True  # Boolean specifies training mode
ALPHA = 0.00003     # step size



print(benchmark(1000))

counter = 0
keep_running = False
_begintime = time.time()
STEPS = 1000 * 5
while counter == 0 or keep_running:
    begintime = time.time()
    counter += 1
    print("Cycle " + str(counter))
    print("Previously processed: " + str(len(all_actions)))
    for game in range(STEPS):
        board_position_log, action_log, gains_log, probs_log, result = play_game(sess, training=TRAINING)
        all_boards.append(board_position_log[-1])
        all_actions.append(action_log)
        all_gains.append(gains_log)
        all_probs.append(probs_log)
        if (game + 1) % math.floor(STEPS / 100) == 0 and game > 0:
            tp = math.floor(time.time() - begintime)
            print(str(game) + " / " + str(STEPS) + \
                  " (" + str(math.ceil(game / STEPS * 100)) + "%) time passed: " + str(datetime.timedelta(seconds=tp)) + \
                  " ETA: T-" + str(datetime.timedelta(seconds=math.floor(tp / game * STEPS - tp))) + \
                  " Avg of last set: " + str(sum([sum(x[0]) for x in all_gains[-math.floor(STEPS / 100):]]) / math.floor(STEPS / 100)))
        for player_index in range(2):
            #rewards_log = rewards_calculator(gains_log[player_index], 0.005)
            if result["WinnerIndex"] != player_index:
                rewards_log = [0 for i in probs_log[player_index]]
            else:
                rewards_log = [1 for i in probs_log[player_index]]
            for reward, current_board, action in zip(rewards_log, board_position_log, action_log[player_index]):
                if TRAINING:
                    sess.run(train_step, feed_dict={input_positions:[[x for y in current_board for x in y]] , labels:[action], learning_rate:ALPHA * reward})
    tp = math.floor(time.time() - begintime)
    print("Total processed: " + str(len(all_actions)) + \
          " Just processed: " + str(STEPS) + \
          " Last batch took: " + str(datetime.timedelta(seconds=tp)) + \
          " Games / second: " + str(STEPS / tp) + \
          " Total avg: " + str(sum([len(x) for x in all_actions]) / len(all_actions)))
    save_model(sess, name='model_' + str(BOARD_SIZE[0]) + 'x' + str(BOARD_SIZE[1]) + '.ckpt')



print(benchmark(1000))


