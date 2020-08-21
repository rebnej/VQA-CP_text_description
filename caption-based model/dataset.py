import numpy as np
import torch
from transformers import *
from torch.utils.data import Dataset
import _pickle as cPickle
import os
import json
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import utils
import random


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _create_entry(question, answer, caption, sim_score):
    if None!=answer:
        
        #answerからimage_idとquestion_idを取り除く
        answer.pop('image_id') #answerから'image_id'を取り除きそれを返す
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer,
        'caption' : caption,
        'sim_score' : sim_score
        }
    return entry


def _load_dataset(dataroot, name, label2ans):
    
    question_path = os.path.join(dataroot, '%s_questions.pkl' %(name))
    questions = cPickle.load(open(question_path, 'rb'))
    questions = sorted(questions, key=lambda x: x['question_id'])
    print('the number of questions', len(questions))
    
    #for sentence embedding
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    
    if 'test'!=name[:4]:
        
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        
        entries = []
        
        #train (for captions)
        ins_path_train = 'data/annotations/instances_train2014.json'
        coco_train = COCO(ins_path_train)
        cats_train = coco_train.loadCats(coco_train.getCatIds())
        catIds_train = coco_train.getCatIds(catNms=cats_train)
        imgIds_train = coco_train.getImgIds(catIds=catIds_train)
        imgIds_train = sorted(imgIds_train) #sort imgIds in order of image id
        #print('the number of images: ', len(imgIds))
        # initialize COCO api for caption annotations
        annFile_train = '{}/annotations/captions_train2014.json'.format(dataroot)
        coco_caps_train=COCO(annFile_train)
        
        #val
        ins_path_val = 'data/annotations/instances_val2014.json'
        coco_val = COCO(ins_path_val)
        cats_val = coco_val.loadCats(coco_val.getCatIds())
        catIds_val = coco_val.getCatIds(catNms=cats_val)
        imgIds_val = coco_val.getImgIds(catIds=catIds_val)
        imgIds_val = sorted(imgIds_val) #sort imgIds in order of image id
        #print('the number of images: ', len(imgIds))
        # initialize COCO api for caption annotations
        annFile_val = '{}/annotations/captions_val2014.json'.format(dataroot)
        coco_caps_val = COCO(annFile_val)
        
        
        cnt = 0
        for question, answer in tqdm(zip(questions, answers)):
        
            coco_name = question['coco_split']
            if coco_name == 'train2014':
                #print('use train')
                coco = coco_train
                imgIds = imgIds_train
                coco_caps = coco_caps_train
            else:
                #print('use val')
                coco = coco_val
                imgIds = imgIds_val
                coco_caps = coco_caps_val
   
            
            img_id = question['image_id']
            imgIds = coco.getImgIds(imgIds = img_id)
            img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            
            capIds = coco_caps.getAnnIds(imgIds=img['id']) #get annotation ids that satisfy given filter conditions
            captions = coco_caps.loadAnns(capIds)
            #print(captions)
            
            #random selection
            #i = random.randint(0, 4)
            #caption = captions[i]['caption']
            
            #sentence embedding and select the hightest one
            sentences = [question['question']]
            for i in range(5):
                sentences.append(captions[i]['caption'])
            
            sentence_embeddings = model.encode(sentences)
            maxsim = 0
            maxid = 1
            for i in range(5):
                cosim = cos_sim(sentence_embeddings[0], sentence_embeddings[i+1])
                if cnt < 5:
                        print('cosim and cap')
                        print(cosim)
                        print(sentences[i+1])
                if cosim > maxsim:
                    maxsim = cosim
                    maxid = i + 1
            caption = sentences[maxid]
            
           
            entries.append(_create_entry(question, answer, caption, maxsim))
            
            if cnt < 5:
                print(cnt)
                print(caption)
                print(maxsim)
                print(entries[cnt])
                cnt += 1
            
    else: # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            #if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                #entries.append(_create_entry(img_id2val[img_id], question, None)) #各質問のエントリを作る(q_id, im_id, im, q, ans)
            entries.append(question, caption, None)
            
    print('entries done')
    print(entries[0])
    
    entry_file = os.path.join('data', 'entry_%s.pkl' % name) 
    cPickle.dump(entries, open(entry_file, 'wb'))
    
    
    return entries





class VQABERTDataset(Dataset):
    def __init__(self, name, use_ans, dataroot='data'):
        super(VQABERTDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test']
        
        self.use_ans = use_ans
        
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        entry_path = 'data/entry_%s.pkl' %name
        if os.path.isfile(entry_path):
            print('find the entries')
            self.entries = cPickle.load(open(entry_path, 'rb'))
            print(self.entries[1])
        else:
            print('make the entries')
            self.entries = _load_dataset(dataroot, name, self.label2ans)
            
        print(len(self.entries))

        self.tokenize() #エントリ（全て）にトークン化された質問を追加
        self.tensorize()


    def tokenize(self, max_length=128):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        #if self.use_ans:
            #for entry in tqdm(self.entries):
            
                #cap_q = ' '.join([entry['caption'], entry['question']])
            
                #encoded_dict = self.tokenizer.encode_plus(cap_q, entry['max_ans_str'], add_special_tokens=True, max_length=max_length,
                                                    # pad_to_max_length=True, return_attention_mask=True, return_tensors='pt') #質問とキャプションを前処理

               #entry['encoded_dict'] = encoded_dict #エントリにトencoded_dictを追加
            
        if True:
            print('do not use ans')
            for entry in tqdm(self.entries):
                encoded_dict = self.tokenizer.encode_plus(entry['caption'], entry['question'], add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt') #質問とキャプションを前処理
                
                entry['encoded_dict'] = encoded_dict #エントリにトencoded_dictを追加
        
        
    def tensorize(self):

        for entry in self.entries:
            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    #entryの'answer'に'labels'と'scores'がある
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None


    def __getitem__(self, index):
        entry = self.entries[index] #indexの質問のエントリentry = {'question_id' : question['question_id'],'image_id': question['image_id'],'image': img,'question': question['question'],'answer':answer}

        encoded_dict = entry['encoded_dict'] #トークン化された質問文
        question_id = entry['question_id']
        input_ids = encoded_dict['input_ids']
        input_mask = encoded_dict['attention_mask']
        input_token_type = encoded_dict['token_type_ids']
        answer = entry['answer']

        if None!=answer:
            labels = answer['labels'] #answerにはlabelとscoreがある(両方リスト)
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates) #ansの候補の数分の0を要素にもつtensorを生成
            if labels is not None:
                target.scatter_(0, labels, scores) #candidate answerのうち, labelsのインデックスのところを,対応するscoresの値にして, その他を0にする
            return input_ids, input_mask, input_token_type, target
        else:
            return input_ids, input_mask, input_token_type, question_id


    def __len__(self):
        return len(self.entries)
