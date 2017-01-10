To run:

python hw4-om.py sts_folder=<folder to STS files> corpus=<corpus to use>

both of the arguments are optional, and they default to ./sts_files/ and xinhua-om-small.txt by default.

The STS input files should be tab delimited, if they are not, the script will attempt to fix that through the use of regular expressions.  In the case of the STS.input.SMTeuroparl.txt file, I had to make some modications by hand (due to the upcoming deadline)

I would advice any reviewer to not run this code due tot he length of time it would take to complete.

This program looks for the relevant STS files, and imports them into a pandas dataframe, which is where I store all relevant values.  Once the sentences, and gold standard correlation values are imported, I then stem the sentences, and at this point I then proceed to train the LSA/LSI, LDA and Doc2Vec models.  Due to limits on computation resources (more on this later), I elected to use the smaller corpus which is approximately 50MB.  Even still, the models took extensive amounts of time to train, especially in the case of the LDA model.  

Once the models were trained, I then use the dictionary which contains word to id mappings, to convert the sentences into word-id and counts, and feed those into the models, which return a vector of length num_topics (fun fact, the LSA/LSI default num_topics is 200, but for LDA, the default is only 100).  I then take the returning vectors of two sentences and compute the cosine similarity, which I then re-scale so instead of going from -1 to 1, it goes from 0 to 5 (like the gold-standard values provided).

I store all the similarities in the dataframe, which I then save the relevant segments into STS.output.* files, and furthermore, I save the entire dataframe as a CSV for reference (which should be in the submission).  Furthermore, since I have 15 output files, I decided to call the perl script from within my python script to generate the required output for submission.  I utilized a stack-overflow post to help with calling the perl script from within my python code here https://stackoverflow.com/questions/798413/how-to-call-a-perl-script-from-python-piping-input-to-it


The Pearson correlation for all my outputs are as follows:


lsi MSRpar  Pearson: 0.04592
lda MSRpar  Pearson: 0.28737
doc2vec MSRpar  Pearson: 0.02059
lsi MSRvid  Pearson: 0.43878
lda MSRvid  Pearson: 0.72610
doc2vec MSRvid  Pearson: 0.35150
lsi SMTeuroparl Pearson: 0.33758
lda SMTeuroparl Pearson: 0.41386
doc2vec SMTeuroparl Pearson: 0.17170
lsi answers-forum   Pearson: nan
lda answers-forum   Pearson: nan
doc2vec answers-forum   Pearson: 0.16791
lsi answers-students    Pearson: 0.44307
lda answers-students    Pearson: 0.54287
doc2vec answers-students    Pearson: 0.29952
lsi belief  Pearson: 0.50091
lda belief  Pearson: 0.78818
doc2vec belief  Pearson: 0.39346
lsi headlines   Pearson: 0.29703
lda headlines   Pearson: -0.43655
doc2vec headlines   Pearson: 0.90864
lsi images  Pearson: 0.00015
lda images  Pearson: 0.19038
doc2vec images  Pearson: -0.48160


Issues:

There are a bunch, but the first one that was a head slapper was that some of the sentences were separated by two spaces, and not a tab.  This made the importing of the input files a little difficult, and as a result I had to write a regular expression to convert multiple spaces into a tab. 

I made the mistake of not getting my code finished before trying to utilize the large corpus.  This resulted in hours of wasted time.

The doc2vec model when trained on the larger corpus, filled up my available hard drive space when I tried to save it to a disk.  The LDA model caused memory errors while I attempted to train.  In some cases while working with the large corpus, my python kernel would unexpectedly die.  

I did utilize a stack-overflow source when I was unsure if my dictionary and corpus were of the appropriate format
https://stackoverflow.com/questions/31821821/semantic-similarity-between-phrases-using-gensim

The use of this code was mostly for a sanity check to ensure that the model training parameters were appropriate.

The LDA model returns a very sparse vector, only containing non-zero (I suspect) components of the vector.  This resulted in having to do some post-processing to the returned vector where I was having to pad the vector with zeros in the dimensions that were not returned.

For a couple of sentences, the LSA and LDA models returned null values due to words being submitted that the model had not seen in the training data, specifically the word 'santorini' and 'trashi'.

I decided due to the number of output files, that it would be best if I could run the correlation perl scirpt from within my python code.  I had to do some googling on how to do that, and came across this stack-overflow post  here:
https://stackoverflow.com/questions/798413/how-to-call-a-perl-script-from-python-piping-input-to-it  The post, while 7 years old nearly, was enough to guide me to getting it functional 


Analysis of Output:

It's hard to analyze this output given that I used the much smaller corpus than originally intended.  Doc2Vec has the advantage that it can infer words it has not seen before, so in the answers-forum topic, it was able to infer a vector for the sentences even though it contained words the model did not see while training.

It appears that the LDA model generates the highest Pearson correlation results, however it did also have the longest training time.  Given the wide variety of ranges on the correlation results, I would say the thing that I need is a larger corpus.

This assignment also taught me the value of getting something working on a smaller and more manageable corpus before moving onto a significantly larger one, as well as becoming better with python iterables, so I can learn to stream larger files from disk instead of having to have them in memory.
