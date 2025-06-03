import numpy as np

def save_loss_to_file(epoch_num, loss, filename='loss_history.txt'):
    with open(filename, 'a') as file:
        file.write('epoch #'+ str(epoch_num) + ', loss ' + str(loss) + '\n')
