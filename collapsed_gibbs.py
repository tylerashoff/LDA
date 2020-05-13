import numpy as np
import multiprocessing
import nltk


class ldaGibbs():

    def __init__(self, docs, K, iterations=1000, alpha=0.5, beta=0.5):

        self.docs = docs
        self.K = K
        self.iterations = iterations

        self.alpha = alpha
        self.beta = beta

        self.n_mk = np.zeros([len(docs), K])
        self.n_kt = None
        self.z = {}
        self.vocab = np.array([])
        pass

    def process_docs(self, doc, minLength=2):
        '''
        Pre-process each document and return an nltk.Text object
        '''

        porter = nltk.PorterStemmer()
        corpum = nltk.Text([
            porter.stem(w.lower())
            for w in nltk.word_tokenize(doc)
            if len(w) >= minLength
        ])
        return (corpum)

    def build_corpus(self):
        '''
        build corpus from documents
        '''

        pool = multiprocessing.Pool(8)
        corpus = pool.map(self.process_docs, self.docs)
        self.docs = corpus

        return (corpus)

    def initialize_values(self):
        '''
        initialize values for use in the collapsed gibbs sampler
        '''

        for m in range(len(self.docs)):

            doc = self.docs[m]
            self.z[str(m)] = np.zeros(len(doc))

            for n in range(len(doc)):

                word = doc[n]

                # append new word and augment for new sizes
                if word not in self.vocab:
                    self.vocab = np.append(self.vocab, word)
                    if self.n_kt is not None:
                        self.n_kt = np.hstack(
                            [self.n_kt,
                             np.zeros(self.K).reshape(self.K, 1)])
                        pass
                    else:
                        self.n_kt = np.zeros(self.K).reshape(self.K, 1)
                        pass
                    pass
                # sample topic
                k = np.random.randint(self.K)
                self.z[str(m)][n] = k

                # increment counts
                t = np.where(self.vocab == word)[0][0]
                self.n_mk[m, k] += 1
                self.n_kt[k, t] += 1
                pass
            pass
        pass

    def sample(self, m, t):
        '''
        sample from topic full conditional distribution
        '''

        term1 = (self.n_kt[:, t] + self.beta) / (
            self.n_kt.shape[1] + self.beta * self.K)
        term2 = (self.n_mk[m] + self.alpha) / (
            sum(self.n_mk[m]) + self.alpha * np.sum(self.n_kt, 1))

        p_z = term1 * term2
        p_z /= sum(p_z)

        k = np.argmax(np.random.multinomial(1, p_z))
        return (k)

    def update(self):
        '''
        update values at each iteration of the collapsed gibbs sampler
        '''
        for m in range(len(self.docs)):

            self.doc = self.docs[m]

            for n in range(len(self.doc)):

                word = self.doc[n]
                k = int(self.z[str(m)][n])
                t = np.where(self.vocab == word)[0][0]

                # decrement counts
                self.n_mk[m, k] -= 1
                self.n_kt[k, t] -= 1

                # update topic
                k = self.sample(m, t)
                self.z[str(m)][n] = k

                # increment counts
                self.n_mk[m, k] += 1
                self.n_kt[k, t] += 1
                pass
            pass
        pass

    def main(self, build_corpus=True, verbose=False):

        if build_corpus:
            if verbose:
                print('building corpus')
                pass
            self.build_corpus()
            pass

        if verbose:
            print('initializing values')
            pass
        self.initialize_values()

        if verbose:
            print('running Gibbs Sampler')
            pass
        for i in range(self.iterations):
            if verbose:
                print(str(i) + '/' + str(self.iterations), end='\r')
                pass
            self.update()
            pass

        if verbose:
            print('all done')
            pass
        return (self.z)

    pass


# sample text
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')
docs = newsgroups.data[:20]

lda = ldaGibbs(docs, K=20, iterations=50, alpha=0.1, beta=0.1)
topics = lda.main(verbose=True)

# print out test topics
print(
    np.array([[
        np.unique(topic, return_counts=True)[0][0] for topic in topics.values()
    ], newsgroups.target_names[:20]]).T)
