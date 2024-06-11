import os
import numpy as np
import json
import re
from collections import defaultdict

def make_vocab_question(input_path):
    """creates a text file with vocabulary from the questions"""
    vocab_set = set() # set to store unique words
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = [] 
    datasets = os.listdir(input_path)
    for dataset in datasets:
        with open(input_path + '/' + dataset) as f:
            questions = json.load(f)['questions']
        set_question_length = [None]*len(questions)
        for iquestion, question in enumerate(questions):
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower()) #tokenises question
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)

            set_question_length[iquestion] = len(words)
        question_length += set_question_length
    
    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    with open('vqa_implementation/datasets/vocab_questions.txt', 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])

    print('make vocab for qs')
    print('number of total words of qs : %d' % len(vocab_set))
    print('maximum qs length is: %d' % np.max(question_length))

def make_vocab_answer(input_path, num_answers):
    """makes a dict for vocab of ans and saves into a txt file"""
    answers = defaultdict(lambda : 0)

    datasets = os.listdir(input_path)
    for dataset in datasets:
        with open (input_path + '/' + dataset) as f:
            annotations = json.load(f)['annotations']
        for annotation in annotations:
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\s]", word):
                    continue
                answers[word] += 1
    answers = sorted(answers, key = answers.get, reverse = True)
    assert('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:num_answers-1]

    with open(r'vqa_implementation/datasets/vocab_answers.txt', 'w') as f:
        f.writelines([w+'\n' for w in top_answers])

    print("make vocab for ans")
    print('number of total words of ans : %d' % len(answers))
    print('keep top %d answers into vocab' %num_answers)

qs_dir = r"vqa_implementation/datasets/Questions"
ann_dir = r"vqa_implementation/datasets/Annotations"
num_ans = 1000
make_vocab_question(qs_dir)
make_vocab_answer(ann_dir, num_ans)