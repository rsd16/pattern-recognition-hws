from PIL import Image, ImageDraw, ImageFont
import sys


character_width = 14
character_height = 25
ignore_list = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
punctuations_list = ['(', ')', ',', '"', '.', '-', '!', '?', '\'']
total_transitions_per_char = {}
transition_prob = {}
initial_prob = {}
total_initial_chars = 0
total_chars = 0
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(filename):
    img = Image.open(filename)
    px = img.load()
    (x_size, y_size) = img.size
    result = []
    for x_beg in range(0, int(x_size / character_width) * character_width, character_width):
        result += [[''.join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + character_width)]) for y in range(0, character_height)]]

    return result

def load_training_letters(filename):
    global TRAIN_LETTERS
    letter_images = load_letters(filename)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

def read_data(fileName):
    file = open(fileName, 'r')
    count = 0
    for line in file:
        line = line.rstrip('\n')
        populate_dict(line.lower())
        count += 1

def populate_dict(line):
    line = remove_pos(line)
    line = remove_punctuation(line)
    global total_chars
    for i in range(0, len(line)):
        total_chars += 1
        if i == 0:
            previous_char = line[i]
            update_initial_prob(line[i])
        else:
            sequence = previous_char + '->' + line[i]
            update_transition_prob(sequence, previous_char)
            previous_char = line[i]

def update_transition_prob(sequence, previous_char):
    if sequence in transition_prob:
        count = transition_prob.get(sequence) + 1
        transition_prob[sequence] = count
    else:
    	transition_prob[sequence] = 1

    if previous_char in total_transitions_per_char:
    	total_transitions_per_char[previous_char] = total_transitions_per_char.get(previous_char) + 1
    else:
    	total_transitions_per_char[previous_char] = 1

def update_initial_prob(ch):
    global total_initial_chars
    total_initial_chars += 1
    if ch in initial_prob:
        initial_prob[ch] = initial_prob.get(ch) + 1
    else:
        initial_prob[ch] = 1

def simple_bayes(train_letters, test_letters):
    simple_bayes_sequence = 'Simple: '
    for i in range(0, len(test_letters)):
        test_char = test_letters[i]
        max_prob = 0
        for key in train_letters:
            hit_count = 1
            miss_count = 1
            space_count = 0
            train_char = train_letters.get(key)
            for j in range(0, len(train_char)):
                for k in range(0, len(train_char[j])):
                    if test_char[j][k] == ' ' and test_char[j][k] == train_char[j][k]:
                        space_count += 1
                    elif test_char[j][k] == '*' and test_char[j][k] == train_char[j][k]:
                        hit_count += 1
                    else:
                        miss_count += 1

            if space_count > ((character_width * character_height) - 5):
                probable_char = key
                break

            if (float(hit_count) / miss_count) > max_prob:
                max_prob = (float(hit_count) / miss_count)
                probable_char = key

        simple_bayes_sequence += probable_char

    return simple_bayes_sequence

def hmm_using_ve(train_letters, test_letters):
    final_sequence = 'HMM VE: '
    prob_char = ''
    tow1 = {}
    tow2 = {}
    total_sum = 0
    for key in total_transitions_per_char:
        total_sum += total_transitions_per_char.get(key)

    for i in range(0, len(test_letters)):
        max_prob = 0
        test_char = test_letters[i]
        if i == 0:
            for j in range(0, len(TRAIN_LETTERS)):
                char_count = 1
                prob = find_emission_prob_per_char(test_char, train_letters.get(TRAIN_LETTERS[j]))
                if prob > max_prob:
                    max_prob = prob
                    prob_char = TRAIN_LETTERS[j]

                if TRAIN_LETTERS[j].lower() in initial_prob:
                    char_count = initial_prob.get(TRAIN_LETTERS[j].lower())

                tow1[TRAIN_LETTERS[j]] = prob * (float(char_count)/total_initial_chars)

            final_sequence += prob_char
        else:
            for j in range(0, len(TRAIN_LETTERS)):
                prob = 0
                total_count = total_sum
                current_letter = TRAIN_LETTERS[j].lower()
                for k in range(0, len(TRAIN_LETTERS)):
                    char_count = total_chars
                    total_count = total_chars
                    previous_letter = TRAIN_LETTERS[k].lower()
                    seq = previous_letter + '->' + current_letter
                    if seq in transition_prob:
                        char_count += transition_prob.get(seq)

                    if previous_letter in total_transitions_per_char:
                        total_count += total_transitions_per_char.get(previous_letter)

                    prob += ((float(char_count) / total_count) * tow1.get(TRAIN_LETTERS[i]) * find_emission_prob_per_char(test_char, train_letters.get(TRAIN_LETTERS[j])))

                if prob > max_prob:
                    max_prob = prob
                    prob_char = TRAIN_LETTERS[j]

                tow2[TRAIN_LETTERS[j]] = prob

            final_sequence += prob_char
            tow1 = tow2.copy()
            tow2.clear()

    return final_sequence

def find_emission_prob_per_char(test_char, train_char):
    hit_count = 1
    space_count = 0
    miss_count = 1
    for j in range(0, len(train_char)):
        for k in range(0, len(train_char[j])):
            if test_char[j][k] == ' ' and test_char[j][k] == train_char[j][k]:
                space_count += 1
            elif test_char[j][k] == '*' and test_char[j][k] == train_char[j][k]:
                hit_count += 1
            else:
                miss_count += 1

    if space_count > ((character_width * character_height) - 5):
        prob = float(space_count) / (character_width * character_height)
    else:
    	prob = float(hit_count) / (character_width * character_height)

    return prob

def hmm_using_viterbi(train_letters, test_letters):
    final_sequence = 'HMM MAP: '
    prob_char = ''
    tow1 = {}
    tow2 = {}
    total_sum = 0
    for key in total_transitions_per_char:
        total_sum += total_transitions_per_char.get(key)

    for i in range(0, len(test_letters)):
        max_prob = 0
        test_char = test_letters[i]
        if i == 0:
            for j in range(0, len(TRAIN_LETTERS)):
                char_count = 1
                prob = find_emission_prob_per_char(test_char, train_letters.get(TRAIN_LETTERS[j]))
                if prob > max_prob:
                    max_prob = prob
                    prob_char = TRAIN_LETTERS[j]

                if TRAIN_LETTERS[j].lower() in initial_prob:
                    char_count = initial_prob.get(TRAIN_LETTERS[j].lower())

                tow1[TRAIN_LETTERS[j]] = prob * (float(char_count) / total_initial_chars)

            final_sequence += prob_char
        else:
            for j in range(0, len(TRAIN_LETTERS)):
                prob = 0
                total_count = total_sum
                current_letter = TRAIN_LETTERS[j].lower()
                for k in range(0, len(TRAIN_LETTERS)):
                    previous_letter = TRAIN_LETTERS[k].lower()
                    char_count = total_chars
                    total_count = total_chars
                    seq = previous_letter + '->' + current_letter
                    if seq in transition_prob:
                        char_count += transition_prob.get(seq)

                    if previous_letter in total_transitions_per_char:
                        total_count += total_transitions_per_char.get(previous_letter)

                    prob = max(prob, ((float(char_count) / total_count) * tow1.get(TRAIN_LETTERS[k])))

                prob *= find_emission_prob_per_char(test_char, train_letters.get(TRAIN_LETTERS[j]))
                if prob > max_prob:
                    max_prob = prob
                    prob_char =  TRAIN_LETTERS[j]

                tow2[TRAIN_LETTERS[j]] = prob

            final_sequence += prob_char
            tow1 = tow2.copy()
            tow2.clear()

    return final_sequence

def remove_pos(line):
    for i in range(0, len(ignore_list)):
        line = line.replace(' ' + ignore_list[i].lower(), '')

    return line

def remove_punctuation(line):
    for i in range(0, len(punctuations_list)):
        line = line.replace(punctuations_list[i] + ' ' + '.' + ' ', punctuations_list[i])

    return line

train_letters = load_training_letters('courier-train.png')
test_letters = load_letters('test-14-0.png')
read_data(str('tra.txt'))
print(simple_bayes(train_letters, test_letters))
print(hmm_using_ve(train_letters, test_letters))
print(hmm_using_viterbi(train_letters, test_letters))

print(total_transitions_per_char)

# Each training letter is now stored as a list of characters, where black dots are represented by *'s and white dots
# are spaces. For example, here's what "a" looks like:
print('\n'.join([r for r in train_letters['a']]))

# Same with test letters. Here's what the third letter of the test data looks like:
print(len(train_letters))
print('\n'.join([r for r in list(train_letters)[2]]))
