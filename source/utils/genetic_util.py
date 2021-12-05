import numpy as np
import selfies as sf


def random_mutation(one_hot, mut_chance=0.2):
    mutation_point = np.random.randint(0, len(one_hot))
    mutation_draw = np.random.random()
    mutation_type = np.random.randint(0, len(one_hot[0]))

    if mutation_draw < mut_chance:
        hot_loc = np.argmax(one_hot[mutation_point])
        one_hot[mutation_point][hot_loc] = 0
        one_hot[mutation_point][mutation_type] = 1

    return one_hot


def multi_mutation(one_hot, mut_steps=5, mut_chance=0.1):
    for _ in range(mut_steps):
        one_hot = random_mutation(one_hot, mut_chance)
    return one_hot


def cross(one_hot_1, one_hot_2):
    temp = []
    temp_2 = []
    cross_point = np.random.randint(0, len(one_hot_1))

    for i in range(len(one_hot_1)):
        if i < cross_point:
            temp.append(one_hot_1[i].tolist())
            temp_2.append(one_hot_2[i].tolist())
        else:
            temp.append(one_hot_2[i].tolist())
            temp_2.append(one_hot_1[i].tolist())

    return np.array(temp), np.array(temp_2)


def draw_from_pop_dist(pop_loss, log = False):
    total_loss = 0
    for i in pop_loss: total_loss += i
    draw = np.random.rand() * total_loss
    ind = 0
    track = pop_loss[0]
    #print(pop_loss)
    while track < draw:
        ind += 1
        track += pop_loss[ind]

    return ind