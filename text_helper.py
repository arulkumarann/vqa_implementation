import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def tokenize(sentence):
    """ splits the sentence into tokens with regex"""
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def load_str_list(fname):
    """ loads a list of strings from a file"""
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines ]
    return lines

class VocabDict:
    def __init__(self, vocab_file):
        """initializes the vocabdict with a vocabfile"""
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w:n_w for n_w, w in enumerate(self.word_list)}
        self.vocab_size= len(self.word_list)
        self.unk2idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None
    
    def idx2word(self, n_w):
        """retrieves a word form its index"""
        return self.word_list[n_w]

    def word2idx(self, w):
        """returns the index of word. if the word is not found returns the index of '<unk>'. if '<unk>' is not 
        defined then raises an error"""
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            raise ValueError(f"word{w} not in dictionary WHILE DICTIONARY DOES NOT CONTAIN '<unk>'")
    
    def tokenize_and_index(self, sentence):
        """
        tokenizes a sentence and converts the tokens to corresponding indices
        """
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds

