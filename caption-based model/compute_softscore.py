"""
This code is slightly modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import sys
import json
import numpy as np
import re
import _pickle as cPickle
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from dataset import Dictionary
import utils


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)") #数字以外.数字以外
comma_strip = re.compile("(\d)(\,)(\d)") #数字,数字
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


# Notice that VQA score is the average of 10 choose 9 candidate answers cases
# See http://visualqa.org/evaluation.html
#score = get_score(answer_count[answer])
def get_score(occurences):
    if occurences == 0:
        return .0
    elif occurences == 1:
        return .3
    elif occurences == 2:
        return .6
    elif occurences == 3:
        return .9
    else:
        return 1.

#!!!!!結局何がしたいの?
#process_punctuation(answer=most frequent ground-truth answer)
def process_punctuation(inText):
    outText = inText
    for p in punct: #punct = [';', r"/", '[', ']', '"', '{', '}',.....
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(comma_strip, inText) != None): #数字,数字がinTextの中にあるならば
            outText = outText.replace(p, '')  #そのpunctの中の記号を消す
        else:
            outText = outText.replace(p, ' ') #そのpunctの中の記号を空白に置き換える
    outText = period_strip.sub("", outText, re.UNICODE) #数字以外.数字以外の部分の.を消す
    return outText #整形したmost frequent ground-truth answerを返す


#answer = process_digit_article(process_punctuation(answer = most frequent ground-truth answer)) (1つの回答)
def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split() #整形された回答を小文字にして単語ごとに分けたリストを返す
    for word in tempText:
        word = manual_map.setdefault(word, word) #manual_map = { 'none': '0', 'zero': '0', 'one': '1',...,  基本的にkeyがwordのvalueを返すが,keyが見つからない場合第二引数を値とし第一引数をkeyとする要素を挿入する
        if word not in articles: #wordがarticles = ['a', 'an', 'the']になかったら(じゃなかったら)
            outText.append(word) #outTextにwordを追加
        else:
            pass
    for wordId, word in enumerate(outText):  #'a','an','the'以外の回答の単語ならば
        if word in contractions:     #contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", ...
            outText[wordId] = contractions[word]  # contractionsにwordがあれば整形して元の場所にいれる
    outText = ' '.join(outText) #outTextの中の文字列を1つの文字列(間は空白)にして返す
    return outText #結局 most frequent ground-truth answerを整形したものを返す


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


#gtruth = preprocess_answer(gtruth = most frequent ground-truth answer)
def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer)) #most frequent ground-truth answerを整形したものを返す
    answer = answer.replace(',', '')
    return answer


#occurence = filter_answers(answers, 9), answers = train_answers + val_answers
def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = ans_entry['multiple_choice_answer'] # most frequent ground-truth answer
        gtruth = preprocess_answer(gtruth) #most frequent ground-truth answerを整形したものを返す
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence #9回以上出現する回答をkey,その


def create_ans2label(occurence, name, cache_root='data/cache'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    cPickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    cPickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


#compute_target(train_answers, ans2label, 'train')
def compute_target(answers_dset, ans2label, name, cache_root='data/cache'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    #各質問に対して処理を行う
    for ans_entry in answers_dset:
        answers = ans_entry['answers'] #その質問に対する10このanswer
        
        answer_count = {}
        for answer in answers:
            #answer_ = answer['answer']
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1 #回答の各単語についてカウントする

        labels = []
        scores = []
        for answer in answer_count: #keyを取り出す(回答のこと)
            if answer not in ans2label: #回答がans2label(辞書)になかったら
                continue
            labels.append(ans2label[answer]) #あったらlabel(対応する数字)(ans2labelの要素は, boats 835, みたいな感じになってる)
            score = get_score(answer_count[answer]) #その回答の頻度による信頼度のスコアを返す
            scores.append(score) #回答(10こ)の異なる回答の種類に対応するスコア
        
        
        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels, #その質問に対する回答(min:1こ, max:10こ)をラベル化したもののリスト
            'scores': scores #上の各回答に対する信頼度のスコア
        })

    utils.create_dir(cache_root) #cache_root='data/cache'ディレクトリを生成
    cache_file = os.path.join(cache_root, name+'_target.pkl') #例えばcache_root='data/cache/train_target.pkl'など
    cPickle.dump(target, open(cache_file, 'wb')) #targetエントリを'data/cache/train_target.pkl'に保存
    return target #各質問への回答(10こまとめて)に対するtargetエントリを返す


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question


if __name__ == '__main__':
    #train_answer_file = 'data/v2_mscoco_train2014_annotations.json'
    #train_answers = json.load(open(train_answer_file))['annotations']

    #val_answer_file = 'data/v2_mscoco_val2014_annotations.json'
    #val_answers = json.load(open(val_answer_file))['annotations']

    #train_question_file = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
    #train_questions = json.load(open(train_question_file))['questions']

    #val_question_file = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
    #val_questions = json.load(open(val_question_file))['questions']

    #answers = train_answers + val_answers #リストの結合
    #occurence = filter_answers(answers, 9)
    
    #there are not val set in VQA-CP dataset
    train_answer_file = 'data/vqacp/VQA-CP data/vqa/vqacp2/raw/annotations/vqacp_v2_train_annotations.json'
    train_answers = json.load(open(train_answer_file))
    
    train_question_file = 'data/vqacp/VQA-CP data/vqa/vqacp2/raw/annotations/vqacp_v2_train_questions.json'
    train_questions = json.load(open(train_question_file))
    
    test_answer_file = 'data/vqacp/VQA-CP data/vqa/vqacp2/raw/annotations/vqacp_v2_test_annotations.json'
    test_answers = json.load(open(test_answer_file))
    
    test_question_file = 'data/vqacp/VQA-CP data/vqa/vqacp2/raw/annotations/vqacp_v2_test_questions.json'
    test_questions = json.load(open(test_question_file))
    
    split = False
    
    if split == True:
        print('split train dataset')
        train_answers, val_answers, train_questions, val_questions = train_test_split(train_answers, train_questions,                                                                                                 test_size=0.3, random_state=0)
    
        train_a_file = os.path.join('data', 'train_answers.pkl') #
        cPickle.dump(train_answers, open(train_a_file, 'wb'))

        train_q_file = os.path.join('data', 'train_questions.pkl') #
        cPickle.dump(train_questions, open(train_q_file, 'wb'))

        val_a_file = os.path.join('data', 'val_answers.pkl') #
        cPickle.dump(val_answers, open(val_a_file, 'wb'))

        val_q_file = os.path.join('data', 'val_questions.pkl') #
        cPickle.dump(val_questions, open(val_q_file, 'wb'))
        
        test_a_file = os.path.join('data', 'test_answers.pkl') #
        cPickle.dump(test_answers, open(test_a_file, 'wb'))

        test_q_file = os.path.join('data', 'test_questions.pkl') #
        cPickle.dump(test_questions, open(test_q_file, 'wb'))

        answers = train_answers + val_answers + test_answers
        occurence = filter_answers(answers, 9)

        cache_path = 'data/cache/trainval_ans2label.pkl'
        #if os.path.isfile(cache_path):
            #print('found %s' % cache_path)
            #ans2label = cPickle.load(open(cache_path, 'rb'))
        #else:
            #ans2label = create_ans2label(occurence, 'trainval')
        
        ans2label = create_ans2label(occurence, 'trainval')
        print('ans2label done')

        compute_target(train_answers, ans2label, 'train') #各質問への回答(10こまとめて)に対するtargetエントリを返す
        compute_target(val_answers, ans2label, 'val')
        compute_target(test_answers, ans2label, 'test')
        
        
    else:
        print('do not split train dataset')
        
        train_a_file = os.path.join('data', 'train_answers.pkl') #
        cPickle.dump(train_answers, open(train_a_file, 'wb'))

        train_q_file = os.path.join('data', 'train_questions.pkl') #
        cPickle.dump(train_questions, open(train_q_file, 'wb'))
        
        val_a_file = os.path.join('data', 'val_answers.pkl') #
        cPickle.dump(test_answers, open(val_a_file, 'wb'))

        val_q_file = os.path.join('data', 'val_questions.pkl') #
        cPickle.dump(test_questions, open(val_q_file, 'wb'))
        
        answers = train_answers + test_answers
        occurence = filter_answers(answers, 9)
        
        cache_path = 'data/cache/trainval_ans2label.pkl'
        
        ans2label = create_ans2label(occurence, 'trainval')
        print('ans2label done')

        compute_target(train_answers, ans2label, 'train') #各質問への回答(10こまとめて)に対するtargetエントリを返す
        compute_target(test_answers, ans2label, 'val')
        
