from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk.data
import re

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')


def RemoveHttp(text_str):
    httpPattern = '[a-zA-z]+://[^\s]*'
    return re.sub(httpPattern, ' ', str(text_str))


def RemoveTag(text_str, key):
    keys = key.split("/")
    patterns = [k + '-[0-9]*' for k in keys]
    pattern = ""
    for ip in range(len(patterns)):
        if ip == 0:
            pattern = patterns[ip]
        else:
            pattern = pattern + "|" + patterns[ip]
    return re.sub(pattern, ' ', str(text_str))


def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', str(text))


def RemoveGit(text_str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str(text_str))


def textProcess(text, key):
    final = []
    text = RemoveHttp(text)  # 移除链接
    if "#" in key:
        text = re.sub('#-[0-9]+', ' ', text)
    else:
        text = RemoveTag(text, key)
    text = RemoveGit(text)  # 移除git链接
    sentences = tokenizer.tokenize(text)  # 分词
    for sentence in sentences:
        sentence = clean_en_text(sentence)  # keep English, digital and space
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word.lower() not in stopwords.words('english')]  # 去除停用词
        for word in word_tokens:
            if word in stopwords.words('english'):
                continue
            else:
                final.append(str(stemmer.stem(word))) # 提取词干
    if len(final) == 0:
        text = ' '
    else:
        text = ' '.join(final)
    return text


def codeMatch(word):
    identifier_pattern = r'''[A-Zz-z]+[0-9]*_.*
                            |[A-Za-z]+[0-9]*[\.].+
                            |[A-Za-z]+.*[A-Z]+.*
                            |[A-Z0-9]+
                            |_ +[A-Za-z0-9]+.+
                            |[a-zA-Z]+[:]{2,}.+   
                            '''
    identifier_pattern = re.compile(identifier_pattern)
    if identifier_pattern.match(word):
        return True
    else:
        return False


def diffProcess(text):
    identifiers = []
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            if codeMatch(word):
                identifiers.append(word)
    if len(identifiers) == 0:
        text = ' '
    else:
        text = ' '.join(identifiers)
    return text
