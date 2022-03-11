
import pickle
from tqdm import tqdm
import json

trainset_path = "data/laic_data/train.json"

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2id = {"SOS_token":2,"EOS_token":1,"PAD_token":0,'UNK':3}
        self.word2count = {}
        self.id2word = {2: "SOS_token", 1: "EOS_token", 0:"PAD_token",3:'UNK'}
        self.n_words = 4  # Count SOS and EOS
        self.symol = ['，','？','《','》','【','】','（','）','、','。','：','；']
    def addSentence(self, sentence):
        for word in sentence.strip().split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word=='':
            return
        if word in self.symol:
            return 
        if word not in self.word2id :
            self.word2id[word] = self.n_words
            self.word2count[word] = 1
            self.id2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1  

enc_vocab=Voc("enc")
rat1_vocab=Voc("rat_1")
rat2_vocab=Voc("rat_2")

with open(trainset_path) as f:
    for line in tqdm(f.readlines()):
        json_obj = json.loads(line)
        enc_vocab.addSentence(json_obj["fact"])
        rat1_vocab.addSentence(json_obj["rat_1"])
        rat2_vocab.addSentence(json_obj["rat_2"])

with open("pkl/laic/encoder_vocab.pickle","wb") as f:
    pickle.dump(enc_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("pkl/laic/rat_1_vocab.pickle","wb") as f:
    pickle.dump(rat1_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("pkl/laic/rat_2_vocab.pickle","wb") as f:
    pickle.dump(rat2_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)