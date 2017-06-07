import logging
import os
from gensim import corpora,models, similarities
import nltk
import csv
import numpy as np
import scipy.sparse
import time
from sklearn.preprocessing import normalize
from six import iteritems
from scipy.sparse import coo_matrix, vstack
from scipy.sparse import csr_matrix
from collections import defaultdict

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
columns = defaultdict(list)

result = []
#coherence_score = []


def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])
    return mat


def prune_least_relevant(s_r,t_r,s_id,t_id,mtx,c_m):
    j = 0
    r_st = s_r[t_id][0]
    r_ts = t_r[s_id][0]
    while j < mtx.get_shape()[0]:
        if ((s_r[j][0] < r_st) or (t_r[j][0] < r_ts)):
            c_m.pop(j)
            mtx = delete_row_csr(mtx,j)
            s_r = delete_row_csr(s_r,j)
            t_r = delete_row_csr(t_r,j)
            j = j - 1
        j = j + 1
    return [mtx, s_r,t_r,c_m]

def redundant_(mtx,c_m,s_ind,t_ind):
    j = s_ind-1
    while j >= 0:
        c_m.pop(0)
        mtx = delete_row_csr(mtx,0)
        j = j - 1
        
    
    while mtx.get_shape()[0] != t_ind-s_ind+1:
        c_m.pop(t_ind-s_ind+1)
        mtx = delete_row_csr(mtx,t_ind-s_ind+1)
        
    return [c_m,mtx]


def rec(mtx,s_ind,t_ind,c_m,url_link):
    
    list_1 = redundant_(mtx,c_m,s_ind,t_ind)
    c_m = list_1[0]
    mtx = list_1[1]
    if len(c_m) <= 2:
        #print "done"
        return 
    else:
        s_r = relevance_score(0,mtx)
        t_r = relevance_score(mtx.get_shape()[0]-1,mtx)

        
        list_ = prune_least_relevant(s_r,t_r,0,mtx.get_shape()[0]-1,mtx,c_m)
        mtx = list_[0]
        s_r = list_[1]
        t_r = list_[2]
        c_m = list_[3]
        
        #selecting best articles
        a_ind =  -1 
        argmax = 0.0
        for j in range(1,s_r.get_shape()[0]-1):
            if argmax < (s_r[j][0]*t_r[j][0]):
                argmax = s_r[j][0]*t_r[j][0]
                a_ind = j
        if a_ind != -1:
            a_id = c_m[a_ind]
            result.append(a_id)
            mtx_cpy = csr_matrix(mtx)
            c_m_cpy = list(c_m)
            rec(mtx,0,a_ind,c_m,url_link)
            mtx = csr_matrix(mtx_cpy)
            c_m=list(c_m_cpy)
            rec(mtx,a_ind,mtx.get_shape()[0]-1,c_m,url_link)
            mtx = csr_matrix(mtx_cpy)
            c_m=list(c_m_cpy)
    
    
    #print result    



def story_chain(mtx,url_link):
    c_m = range(0,12111)
    s = raw_input("Enter the start node")
    t = raw_input("Enter the end node")
    #s = "http://www.thehindu.com/todays-paper/tp-national/tp-otherstates/congress-campaign-to-counter-anticorruption-movement/article2100277.ece"
    #t = "http://www.thehindu.com/business/Industry/Nestle-India%E2%80%99s-fourth-quarter-profit-hit-by-demonetisation/article17308187.ece"
    print s,t
    s_id = url_link.index(s)
    t_id = url_link.index(t)
    rec(mtx,s_id,t_id,c_m,url_link) 
    result.append(s_id)
    result.append(t_id)
    result.sort()
    for i in range(0,len(result)):
        print url_link[result[i]]
        #print result[i]
    #for i in range(0,len(coherence_score)):
     #   print coherence_score[i]
    #print min(coherence_score)


def relevance_score(k,mtx):
    # normalizing col matrix for mtx
    mtx_normalized = normalize(mtx, norm='l1', axis=0)
    #normalizing col matrix for mtx_t
    mtxt_normalized = normalize(scipy.sparse.csr_matrix.transpose(mtx), norm='l1', axis=0)
    print mtxt_normalized
    row = [k]
    col = [0]
    data = [1]
    c = 0.15
    u = scipy.sparse.csr_matrix((data, (row, col)), shape=(mtx.get_shape()[0]+mtx.get_shape()[1], 1))
    a = u.multiply(c)
    count = 0
    while True:
        ui = u
        e = ui[mtx.get_shape()[0]:mtx.get_shape()[0]+mtx.get_shape()[1],:]
        w = ui[0:mtx.get_shape()[0],:]      
        m1 = mtx_normalized * e #k:k-1+n
        m2 = mtxt_normalized * w #0:k-1
        m = vstack([m1,m2]).todense()
        m = scipy.sparse.csr_matrix(m)
        s = m.multiply(1-c)
        u = s + a
        diff = u-ui 
        det_u = np.sqrt((diff.data*diff.data).sum(0))
        count = count + 1
        if det_u <= 0.000000001:
            break

    return u[0:mtx.get_shape()[0],:]

    


def set_to_store():
    with open('datasets1.csv') as f:      #to extract values from each col ('all the articles')
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v)

def pre():
    set_to_store()
    dictionary = corpora.Dictionary(Articles.lower().split() for Articles in columns['Articles'])
    dictionary.compactify()
    dictionary.save('/tmp/deerwester.dict')
   # print(dictionary)

    class MyCorpus(object):
        def __iter__(self):
            for Articles  in columns['Articles']:
                 # assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(Articles.lower().split())

    corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus_memory_friendly)
    print(corpus_memory_friendly)
    #for vector in corpus_memory_friendly:  # load one vector into memory at a time
     #   print(vector)

    if (os.path.exists("/tmp/deerwester.dict")):
        dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
        corpus = corpora.MmCorpus('/tmp/deerwester.mm')
        print("Used files generated from first tutorial")
    else:
        print("Please run first tutorial to generate data set")


    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]
    i = 0
    j = 0
    row = []
    col = []
    data = []
    for doc,url in zip(corpus_tfidf,columns['Url']):
        url_link = list(map(lambda i: i, columns['Url']))
        for id,value in doc:
            #print i,id,value
            row.append(i)
            col.append(id)
            data.append(value)
        i = i + 1

    mtx = scipy.sparse.csr_matrix((data, (row, col)), shape=(12111, 85809))
    #print mtx
    story_chain(mtx,url_link)
    
    
    
pre()
print("--- %s seconds ---" % (time.time() - start_time))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            