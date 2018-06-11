##Begin codeAIgaming
botName = 'yourbotname'

from random import randint
from random import choice
import tensorflow as tf
import numpy as np
import time
import datetime
import math
import requests
import os

session = None

##googleDriveDownload
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

    return destination

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

##codeInitialise
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

##codeLoad
def load_model(name='model.ckpt'):
    global session
    tf.reset_default_graph()
    initialise_tf()
    
    session = tf.Session()
    tf.train.Saver().restore(session, name)
    print("------------ Model restored. ------------")

##codeGuessMove
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

##codeMainFunction
def calculateMove(gamestate):
    global action_log, inputs_units, hidden_units, output_units, session, model_name, RANDOM_PLACEMENT, BOARD_SIZE, SHIPS
    
    print(gamestate)
    
    if not os.path.isfile('/tmp/checkpoint') or session is None:
        session = None

        BOARD_SIZE = [8, 8]        
        model_name = '/tmp/model_8x8.ckpt'
        links = [
            {"name": "/tmp/model_8x8.ckpt.meta", "id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, 
            {"name": "/tmp/model_8x8.ckpt.index", "id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, 
            {"name": "/tmp/model_8x8.ckpt.data-00000-of-00001", "id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, 
            {"name": "/tmp/checkpoint", "id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
            ]

        for link in links:
            print("finished downloading:", download_file_from_google_drive(link["id"], link["name"]))

        inputs_units = BOARD_SIZE[0] * BOARD_SIZE[1]
        hidden_units = BOARD_SIZE[0] * BOARD_SIZE[1]
        output_units = BOARD_SIZE[0] * BOARD_SIZE[1]

    if session is None:
        load_model(model_name)

    current_board = [[-1 if cell == -1 else 1 if cell == gamestate["Role"] else 0 for cell in row] for row in gamestate["Board"]]
    place_index, _ = guess_move(current_board, gamestate["PossibleMoves"], False)
    place = [int(place_index / BOARD_SIZE[1]), place_index % BOARD_SIZE[1]]
    place = {"Row": int(place[0]), "Column": int(place[1])}
    return place
##End codeAIgaming