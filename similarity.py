import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import os
import sys
from gensim import corpora, models, similarities
import numpy as np
from scipy.spatial.distance import cosine
import subprocess

# For doc2vec
from gensim.models.doc2vec import TaggedLineDocument

# Used this post as an example of a proper way to build a dictionary and corpus
# https://stackoverflow.com/questions/31821821/semantic-similarity-between-phrases-using-gensim

#Used the following post to determine how to run a Perl script within python
# https://stackoverflow.com/questions/798413/how-to-call-a-perl-script-from-python-piping-input-to-it

def file_check(sts_folder):
    files = os.listdir(sts_folder)
    sts_input_files = []
    sts_gs_files = []
    for f in files:
        if 'input' in f.split('.'):
            sts_input_files.append(f)
            # pandas dataframe input requires tab delimited data
            with open(sts_folder + f, 'r+') as _file:
                file_contents = _file.read()
                file_contents = re.sub('  +\n', '\n', file_contents)
                file_contents = re.sub('\t+\n', '\n', file_contents)
                file_contents = re.sub("\.  +", "\t", file_contents)
                file_contents = re.sub('\t\t+', '\t', file_contents)
                _file.seek(0)
                _file.write(file_contents)
        elif 'gs' in f.split('.'):
            sts_gs_files.append(f)

    if len(sts_input_files) == 0:
        print('No Input Files Found')
        raise OSError
    if len(sts_gs_files) == 0:
        print('No GS Files Found')
        raise OSError
    return None


def build_dataframe(sts_folder):
    files = sorted(os.listdir(sts_folder))
    gs_files = []
    input_files = []
    for f in files:
        if f.split('.')[1] == 'gs':
            gs_files.append(f)
        elif f.split('.')[1] == 'input':
            input_files.append(f)


    df = pd.DataFrame()
    input_cols = ['s1', 's2']
    gs_col = ['human_sim']
    for input_file, gs_file in zip(input_files, gs_files):
        input_topic = input_file.split('.')[2]
        gs_topic = gs_file.split('.')[2]
        if gs_topic != input_topic:
            raise NameError('File Mismatch for {} and {}'.format(input_file, gs_file))
        else:
            topic = gs_topic
        
        df_new = pd.concat([pd.read_table(sts_folder + input_file,
                                          sep='\t',
                                          names=input_cols,
                                          quoting=3),
                            pd.read_table(sts_folder + gs_file,
                                          sep='\t',
                                          names=gs_col)], axis=1)

        df_new[['s1_prepped', 's2_prepped']] = df_new[['s1', 's2']].applymap(pre_prep_sentences)
        df_new['topic'] = topic
        df = pd.concat([df, df_new], ignore_index=True)
    return df


def pre_prep_sentences(sentence):
    sentence = str(sentence)
    stop_word_tuple = tuple(stopwords.words('english')) 
    tokenizer = TreebankWordTokenizer()
    try:
        filtered_words = [word for word in tokenizer.tokenize(sentence)\
                          if word.lower().decode('utf-8') not in stop_word_tuple \
                          and re.search("^[a-zA-Z]+$", word)]
    except TypeError:
        print('Encountered TypeError on sentence: {}'.format(sentence))
        return(None)
    
    return(' '.join([SnowballStemmer('english').stem(word).lower() \
                     for word in filtered_words]))


def check_for_corpus_and_dict(corpus_fname, docs_fname, dict_fname):
    try:
        f = open(corpus_fname, 'r')
        corpus = []
        for item in f:
            corpus.append(item)
        f.close()
        dictionary = check_for_dictionary(dict_fname, docs_fname)
    except IOError:
        print('Generating Corpus, this may take a while...')
        texts = [[word.lower() for word in document.lower().split()]
                 for document in open(docs_fname)]
    
        dictionary = check_for_dictionary(dict_fname, docs_fname, texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        f = open(corpus_fname, 'w')
        for item in corpus:
            f.write("%s\n" % item)
        f.close()
    return corpus, dictionary


def check_for_dictionary(dict_fname, docs_fname, texts=None):
    try:
        dictionary = corpora.Dictionary.load(dict_fname)
    except IOError:
        print('Generating Dictionary, this may take a while')
        if texts:
            dictionary = corpora.Dictionary(texts)
            once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems()\
             if docfreq == 1]
            dictionary.filter_tokens(once_ids)
            dictionary.compactify()
            dictionary.save(dict_fname)
        else:
            print('No texts provided')
            raise IOError
    return dictionary


def check_for_lsimodel(lsi_fname, corpus, dictionary):
    try:
        lsi = models.LsiModel.load(lsi_fname)
    except IOError:
        print('Training LSI model, this may take some time')
        lsi = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary)
        lsi.save(lsi_fname)
    return lsi


def check_for_ldamodel(lda_fname, corpus, dictionary):
    try:
        lda = models.LdaModel.load(lda_fname)
    except IOError:
        print('Training LDA model, this will take a long time')
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary)
        lda.save(lda_fname)
    return lda


def check_for_doc2vecmodel(doc2vec_fname, docs_fname, corpus, dictionary):
    try:
        doc2vec = models.Doc2Vec.load(doc2vec_fname)
    except IOError:
        print('Training Doc2Vec model, this may take a long time')
        documents = TaggedLineDocument(docs_fname)
        doc2vec = models.doc2vec.Doc2Vec(documents=documents, workers=4)
        doc2vec.save(doc2vec_fname)
    return doc2vec


def get_vector(sentence, dictionary, model):
    vec_bow = dictionary.doc2bow(sentence.lower().split())   
    vector = np.zeros(model.num_topics)
    for component in model[vec_bow]:
        vector[component[0]] = component[1]
    return vector


def calc_cosine_similarity(sentence1, sentence2, dictionary, model):
    vector_1 = get_vector(sentence1, dictionary, model)
    vector_2 = get_vector(sentence2, dictionary, model)
    try:
        cosine_angle = convert_spatial_to_similarity(cosine(vector_1, 
                                                            vector_2))
    except ValueError:
        return np.nan
    return rescale_similarity_scalar(cosine_angle)

# gold standard files ranged from 0 to 5, cosine angles range from -1 to 1
# rescaling for consistent normalization
def rescale_similarity_scalar(cosine_angle):
    return 2.5 + 2.5*(cosine_angle)

#spatial cosine = 1 - udotv / mag(u) * mag(v) so need to convert to cosine similarity
def convert_spatial_to_similarity(spatial_cosine):
    return 1 - spatial_cosine

def calc_doc2vec_similarity(sentence_1, sentence_2, doc2vec):
    s1_inferred = doc2vec.infer_vector(str(sentence_1).split())
    s2_inferred = doc2vec.infer_vector(str(sentence_2).split())
    return rescale_similarity_scalar(\
                convert_spatial_to_similarity(cosine(s1_inferred,\
                                                     s2_inferred)))

# function made possible due to stack-overflow question:
# https://stackoverflow.com/questions/798413/how-to-call-a-perl-script-from-python-piping-input-to-it
def eval_pearson(output_file, sts_folder):
    script_filename = 'correlation-noconfidence.pl'
    script_directory = os.getcwd() + sts_folder[1:]
    perl_script_path = script_directory + script_filename
    topic = output_file.split('.')[2]
    model = output_file.split('.')[3]
    gs_file = script_directory + 'STS.gs.' + topic + '.txt'
    full_output_file = script_directory + output_file
    pipe = subprocess.Popen([perl_script_path, full_output_file, gs_file], 
                            stdout=subprocess.PIPE)
    result = pipe.stdout.read()
    print('\t'.join([model, topic, result.rstrip('\n')]))
    return None


def main(argv):
    try:
        parameters = argv[1:]
    except IndexError:
        pass
    docs_fname = 'xinhua-om-small.txt'
    sts_folder = './sts_files/'

    for parameter in parameters:
        if parameter.startswith('sts_folder'):
            sts_folder = parameter.split('=')[1]
        elif parameter.startswith('corpus'):
            docs_fname = parameter.split('=')[1]

    file_check(sts_folder)
    df = build_dataframe(sts_folder)


    corpus_fname = docs_fname.split('.')[0] + ".corpus"
    dict_fname = docs_fname.split('.')[0] + '.dict'
    lsi_fname = docs_fname.split('.')[0] + '-lsi.model'
    lda_fname = docs_fname.split('.')[0] + '-lda.model'
    doc2vec_fname = docs_fname.split('.')[0] + '-doc2vec.model'

    corpus, dictionary  = check_for_corpus_and_dict(corpus_fname, docs_fname, dict_fname)

    lsi = check_for_lsimodel(lsi_fname, corpus, dictionary)
    lda = check_for_ldamodel(lda_fname, corpus, dictionary)
    doc2vec = check_for_doc2vecmodel(doc2vec_fname, docs_fname, corpus, dictionary)
    df['lsi'] = df.apply(lambda x: calc_cosine_similarity(x['s1_prepped'],
                                                          x['s2_prepped'], 
                                                          dictionary, 
                                                          lsi), axis=1)
    df['lda'] = df.apply(lambda x: calc_cosine_similarity(x['s1_prepped'], 
                                                          x['s2_prepped'], 
                                                          dictionary, 
                                                          lda), axis=1)
    df['doc2vec'] = df.apply(lambda x: calc_doc2vec_similarity(x['s1_prepped'], 
                                                               x['s2_prepped'], 
                                                               doc2vec), axis=1)
    for topic in df.topic.unique():
        df_subtopic = df[df.topic == topic]
        for model in ['lsi', 'lda', 'doc2vec']:
            output_file = 'STS.output.' + topic +'.' + model + '.txt'
            f = open(sts_folder + output_file, 'w+')
            for value in df_subtopic[model].values.tolist():
                f.write(str(value) + '\n')
            f.close()
            eval_pearson(output_file, sts_folder)
    df.to_csv('complete_output.csv')
    return None

if __name__ == "__main__":
    main(sys.argv)
