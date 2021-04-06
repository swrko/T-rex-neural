import multiprocessing
import os
import pickle
import neat
import numpy as np
import dataset_creator

import cv2
import pyautogui
from time import sleep
import _thread as th
from mss import mss


dsc = dataset_creator.DatasetCreator()
inp, tar = dsc.get_rand_dataset()


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0.0
    for i in range(len(inp)):
        out = net.activate(tuple(inp[i]))
        l_out = tuple(tar[i])
        err = [abs(x - y) for x, y in zip(l_out, out)]
        # err = [tuple(((x - y) ** 2.0) / 2.0) for x, y in zip(l_out, out)]
        # err = sum(((l_out - out) ** 2.0) / 2.0)
        fitness += sum(err)
    return 100000.0 - fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_path):
    # Load the config file, which is assumed to live in
    # the same directory as this script.

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 50)

    # Save the winner
    with open('winner-50p', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


def test(file_path, config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    with open(file_path, 'rb') as f:
        winner = pickle.load(f)
    print(winner)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # out = net.activate(tuple([50, 50]))
    # print(out)

    sct = mss()

    while (True):
        screen_np = np.array(sct.grab({'top': 100, 'left': 0, 'width': 600, 'height': 200}))
        # screen_np = np.array(sct.grab({'top': 0, 'left':0, 'width': 600, 'height': 200}))
        inputs = inputs_from_process_img(screen_np)
        outputs = net.activate(tuple(inputs))
        print("outputs: {}".format(outputs))
        th.start_new_thread(decide3, (outputs,))

        if cv2.waitKey(15) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


    # reading inputs from image -> first value lower roi, second value is high roi
    # frames_path = r'C:\Users\Herny\Documents\shady skola\DP\T_rex\frames'
    # im_path = os.path.join(frames_path, '49797_[0, 1, 0].png')
    # image = cv2.imread(im_path)
    # print(inputs_from_process_img(image))



def inputs_from_process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # povodne
    # roiM = processed_img[150, 85:400]
    # roiD = processed_img[175, 85:400]

    roiM = processed_img[100, 85:500]
    roiD = processed_img[120, 85:500]

    # find all black pixels at roi
    firstM = np.where(roiM == 0)
    firstD = np.where(roiD == 0)

    input = np.array([400, 400])

    # if is not empty -> rewrite
    if len(firstD[0]) != 0:
        input[0] = firstD[0][0]
    if len(firstM[0]) != 0:
        input[1] = firstM[0][0]
    # original values
    # cv2.line(processed_img, (90, 155), (400, 155), (0, 0, 0), 1)
    # cv2.line(processed_img, (90, 175), (400, 175), (0, 0, 0), 1)

    # great but comennted for compute power
    cv2.line(processed_img, (85, 100), (500, 100), (0, 0, 0), 1)
    cv2.line(processed_img, (85, 120), (500, 120), (0, 0, 0), 1)
    cv2.imshow('dsad', processed_img)
    return list(input)

def decide(outputs):
    index_of_max = np.argmax(outputs[0])
    value_of_max = outputs[0][index_of_max]
    if index_of_max == 0 and value_of_max >= 0.78:
        pyautogui.keyDown(' ')
        sleep(0.02)
        pyautogui.keyUp(' ')
    elif index_of_max == 1 and value_of_max > 0.5:
        pyautogui.keyDown('down')
        sleep(0.4)
        pyautogui.keyUp('down')
    else:
        pass

def decide2(outputs):
    index_of_max = np.argmax(outputs)
    value_of_max = outputs[index_of_max]
    if index_of_max == 0 and value_of_max >= 0.78:
        pyautogui.keyDown(' ')
        sleep(0.02)
        pyautogui.keyUp(' ')
    elif index_of_max == 1 and value_of_max > 0.65:
        pyautogui.keyDown('down')
        sleep(0.4)
        pyautogui.keyUp('down')
    else:
        pass

def decide3(outputs):
    index_of_max = np.argmax(outputs)
    value_of_max = outputs[index_of_max]
    if index_of_max == 0 and value_of_max >= 0.99:
        pyautogui.keyDown(' ')
        sleep(0.02)
        pyautogui.keyUp(' ')
    elif index_of_max == 1 and value_of_max > 0.99:
        pyautogui.keyDown('down')
        sleep(0.5)
        pyautogui.keyUp('down')
    else:
        pass

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'config-feedforward')
    # run(config_path)

    file_path = os.path.join(local_dir, 'winner-50p')
    test(file_path, config_path)
