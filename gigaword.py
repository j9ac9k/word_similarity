from __future__ import division
from __future__ import print_function
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import re
import os
import gzip
import xml.etree.cElementTree as ET
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import time
import sys


def download_files():
    url = 'http://cslu.ohsu.edu/~wst/nlp-class/hw/hw2_files/data/'
    html = urlopen(url).read()
    regex_expression = "'>(.*?)</a>"
    try:
        file_names = re.findall(regex_expression, html)
    except TypeError:
        file_names = re.findall(regex_expression.encode('utf-8'), html)
    print('Downloading files from ' + url)
    for link in file_names:
        if type(link) == bytes:
            link = link.decode('utf-8')
        print(url + link)
        download_file = urlopen(url + link)
        output = open(link, 'wb')
        output.write(download_file.read())
        output.close()
    print('Download complete')
    return file_names


def find_files():
    file_names = []
    for individual_file in os.listdir(os.curdir):
        if individual_file.endswith(".gz"):
            file_names.append(individual_file)
    if len(file_names) != 173:
        file_names = download_files()
    return sorted(file_names)


def main(argv):
    if 'run_test' in argv:
        run_test = True
    else:
        run_test = False
    if 'print_runtime' in argv:
        print_runtime = True
        start_time = time.time()
    else:
        print_runtime = False
    file_names = find_files()
    stop_word_set = set(stopwords.words('english'))
    if 'short_test' in argv:
        files_to_process =  ['xin_eng_200201.xml.gz']
    else:
        files_to_process = file_names
    tokenizer = TreebankWordTokenizer()
    wonl = WordNetLemmatizer()
    output_filename = 'xinhua-om-lema.txt'
    output = open(output_filename, "wt")
    for downloaded_file in files_to_process:
        print('Working on {}'.format(downloaded_file))
        for paragraph in [paragraphs.text for paragraphs in ET.fromstring(gzip.open(downloaded_file).read())\
                          .findall(".//*[@type='story']//P")]:
            if not paragraph:
                continue
            for sentence in sent_tokenize(paragraph):
                filtered_words = [word for word in tokenizer.tokenize(sentence)\
                                  if word.lower() not in stop_word_set \
                                  and re.search("^[a-zA-Z]+$", word)]
                if not filtered_words:
                    continue
                output.write(' '.join([wnl.lemmatize(word).lower()
                                       for word in filtered_words]) + '\n')
    output.close()
    if print_runtime:
        run_time = time.time() - start_time
        print('Total Processing Time: {0:.2f} minutes'.format(run_time/60))
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
