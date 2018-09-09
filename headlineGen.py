


import pickle 
import json
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers.core import Lambda
import keras.backend as K
import h5py
import Levenshtein
import random, sys
seed=5
np.random.seed(seed)
random.seed(seed)


class HeadlineGen:
    def __init__(self):
        self.dataFile = "signalmedia-1m.jsonl"
        self.embeddingFile = "vocabulary-embedding.pkl"
        self.trainWeightsFile = "train.hdf5"
        self.historyFile = "history.hdf5"
        self.vocabularySize = 40000
        self.embeddingDimension = 100
        self.lower = True
        self.empty = 0
        self.eos = 1
        self.gloveFile = "glove.6B.100d.txt"
        self.matchingThreshold = 0.5
        self.nbUnknownWords = 100
        
        self.maxlenHeads_training = 25
        self.maxlenDescs_training = 25
        self.sequenceLen_training = self.maxlenDescs_training+self.maxlenHeads_training
        self.maxlenHeads_predict = 50
        self.maxlenDescs_predict = 25
        self.sequenceLen_predict = self.maxlenDescs_predict+self.maxlenHeads_predict
        self.lstmUnits = 512
        self.lstmLayers = 3
        self.batchNorm = False
        self.activationLayer = 40
        
        self.optimizer = 'adam'
        self.LR = 1e-4
        self.batchSize = 64
        self.flips = 10
        self.trainSample = 30000
        self.validSample = 3000
        self.pW = 0
        self.pU = 0
        self.pDense = 0
        
        
        
    def vocabularyGen(self):
        self.descriptions = []
        self.headlines = []
        for line in open(self.dataFile, 'r'):
            self.descriptions.append(json.loads(line)['content'])
            self.headlines.append(json.loads(line)['title'])
        
        if self.lower:
            self.headlines = [h.lower() for h in self.headlines]
            self.descriptions = [h.lower() for h in self.descriptions]
        
        vocabcount = Counter(w for txt in (self.headlines+self.descriptions) for w in txt.split())
        vocab = map(lambda x: x[0], sorted(vocabcount.items(), key = lambda x: -x[1]))
        
        self.word2idx = dict((word, idx+self.eos+1) for idx, word in enumerate(vocab))
        self.word2idx['<empty>'] = self.empty
        self.word2idx['<eos>'] = self.eos
        
        self.idx2word = dict((idx,word) for word,idx in self.word2idx.iteritems())
        
        gloveIndexDict = {}
        gloveEmbeddingWeights = np.empty((400000, self.embeddingDimension))
        globalScale = 0.1
        with open(self.gloveFile, 'r') as fp:
            i = 0
            for l in fp:
                l = l.strip().split()
                w = l[0]
                gloveIndexDict[w] = i
                gloveEmbeddingWeights[i,:] = map(float,l[1:])
                i += 1
        
        gloveEmbeddingWeights *= globalScale
        
        for w,i in gloveIndexDict.iteritems():
            w = w.lower()
            if w not in gloveIndexDict:
                gloveEmbeddingWeights[w] = i
        
        embeddingScale = gloveEmbeddingWeights.std()*np.sqrt(12)/2
        self.embeddingMatrix = np.random.uniform(low=-embeddingScale, high=embeddingScale, size=(self.vocabularySize,self.embeddingDimension))
        
        
        for i in range(self.vocabularySize):
            word = self.idx2word[i]
            gloveIndex = gloveIndexDict.get(word, gloveIndexDict.get(word.lower()))
            if (gloveIndex is None) and (word.startswith('#')):
                word = word[1:]
                gloveIndex = gloveIndexDict.get(word, gloveIndexDict.get(word.lower()))
            if gloveIndex is not None:
                self.embeddingMatrix[i,:] = gloveEmbeddingWeights[gloveIndex,:]
        
        word2glove = {}
        for word in self.word2idx:
            if word in gloveIndexDict:
                gloveWord = word
            elif word.lower() in gloveIndexDict:
                gloveWord = word.lower()
            elif ((word.startswith('#')) and (word[1:] in gloveIndexDict)):
                gloveWord = word[1:]
            elif (w.startswith('#')) and (w[1:].lower() in gloveIndexDict):
                gloveWord = word[1:].lower()
            else:
                continue
            word2glove[w] = gloveWord
        
        normEmbeddingMatrix = self.embeddingMatrix/np.array([np.sqrt(np.dot(weights,weights)) for weights in self.embeddingMatrix])[:,None]
        
        match = []
        for word,i in self.word2idx.iteritems():
            if i >= self.vocabularySize-self.nbUnknownWords and w.isalpha() and w in word2glove:
                gloveidx = gloveIndexDict[self.word2glove[w]]
                gweight = gloveEmbeddingWeights[gloveidx,:].copy()
                gweight /= np.sqrt(np.dot(gweight,gweight))
                score = np.dot(normEmbeddingMatrix[:self.vocabularySize-self.nbUnknownWords], gweight)
                while True:
                    embeddingIdx = score.argmax()
                    s = score[embeddingIdx]
                    if s < self.matchingThreshold:
                        break
                    if self.idx2word[embeddingIdx] in word2glove :
                        match.append((word, embeddingIdx, s)) 
                        break
                    score[embeddingIdx] = -1
        match.sort(key = lambda x: -x[2])
        
        self.gloveId2Id = dict((self.word2idx[word], embeddingIdx) for word, embeddingIdx, _ in match)
        
        self.Y = [[self.word2idx[words] for words in heads] for heads in self.headlines]
        
        self.X = [[self.word2idx[words] for words in descs] for descs in self.descriptions]
        
        with open(self.embeddingFile, 'wb') as fp:
            pickle.dump((self.embeddingMatrix, self.idx2word, self.word2idx, self.gloveId2Id), fp, -1)
            
        
    
    
    
    
    def modelGen(self, maxlenDescs, maxlenHeads):
        activationLayer = self.activationLayer
        
        
        def attentionLayer(X, mask, n=activationLayer, maxlenDescs=maxlenDescs, maxlenHeads=maxlenHeads):
            desc, head = X[:,:maxlenDescs,:], X[:,maxlenDescs:,:]
            head_activations, headWords = head[:,:,:n], head[:,:,n:]
            desc_activations, descWords = desc[:,:,:n], desc[:,:,n:]
            activation = K.batch_dot(head_activations, desc_activations, axes=(2,2))
            activation = activation + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlenDescs],'float32'),1)
            activation = K.reshape(activation,(-1,maxlenDescs))
            activationWeights = K.softmax(activation)
            activationWeights = K.reshape(activationWeights,(-1,maxlenHeads,maxlenDescs))
            descAvgWord = K.batch_dot(activationWeights, descWords, axes=(2,1))
            return K.concatenate((descAvgWord, headWords))
        
        model = Sequential()
        model.add(Embedding(self.vocabularySize, self.embeddingDimension, input_length=self.sequenceLen, embeddings_regularizer=None, weights=[self.embeddingMatrix], mask_zero=True,name='embedding_1'))
        model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_1'))
        model.add(Dropout(self.pDense ,name='dropout_1'))
        model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_2'))
        model.add(Dropout(self.pDense ,name='dropout_2'))
        model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_3'))
        model.add(Dropout(self.pDense ,name='dropout_3'))
        model.add(Lambda(activationLayer, mask = lambda inputs, mask: mask[:,maxlenDescs:], output_shape = lambda input_shape: (input_shape[0], maxlenHeads, 2*(self.lstmUnits - activationLayer)), name='attentionLayer_1'))
        model.add(TimeDistributed(Dense(self.vocabularySize, name = 'timedistributed_1')))
        model.add(Activation('softmax', name='activation_1'))


        return model
    
    
    def genSamples(self, model = None, X = None, X_test = None, Y_test = None, avoid = None, avoidScore = 1, skips = 2, k = 10, batchSize = 64, short = True, activation = 1, oov = True):
        maxlenDescs = self.maxlenDescs_predict
        sequenceLen = self.sequenceLen_predict
        vocabularySize = self.vocabularySize
        unknownWords = 10
        outVocabulary = vocabularySize-unknownWords
        eos = self.eos
        empty = self.empty
        
        
        with h5py.File(self.trainWeightsFile, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            weights = [np.copy(v) for v in f['time_distributed_1']['time_distributed_1'].values()]
                
        def leftPadding(x, maxlenDescs=maxlenDescs, eos=eos):
            if maxlenDescs == 0:
                return [eos]
            l = len(x)
            if l > maxlenDescs:
                x = x[-maxlenDescs:]
                l = maxlenDescs
            return([empty]*(maxlenDescs-l) + x + [eos])
        
        
        def outVocabFold(xStart):
            
            xStart = [x if x < outVocabulary else self.gloveId2Id.get(x,x) for x in xStart]
            outside = sorted([x for x in xStart if x >= outVocabulary])
            outside = dict((x,vocabularySize-1-min(i, unknownWords-1)) for i, x in enumerate(outside))
            xStart = [outside.get(x,x) for x in xStart]
            return xStart
        
        def outVocabUnfold(desc,xStart):
            unfold = {}
            for i, unfold_idx in enumerate(desc):
                fold_idx = xStart[i]
                if fold_idx >= outVocabulary:
                    unfold[fold_idx] = unfold_idx
            return [unfold.get(x,x) for x in xStart]
        
        def softmaxLayer(inp):
            inp = np.dot(inp, weights[1]) + weights[0]
            inp -= inp.max()
            inp = np.exp(inp)
            inp /= inp.sum()
            return inp
        
        
        def modelPredict(samples, empty=empty, sequenceLen=sequenceLen):
            samplesLen = map(len, samples)
            
            data = sequence.pad_sequences(samples, maxlen=sequenceLen, value=empty, padding='post', truncating='post')
            probs = model.predict(data, verbose=0, batch_size = batchSize)
            return np.array([softmaxLayer(prob[sampleLength-maxlenDescs-1]) for prob, sampleLength in zip(probs, samplesLen)])
        
        
        def beamSample(predict, start=[empty]*maxlenDescs + [eos], k=1, avoid = None, avoidScore = 1, maxSample = sequenceLen, oov = True, outV = vocabularySize-1 ,empty=empty, eos=eos, activation=1.0):
            
            def sample(score, n, activation=activation):
                n = min(n, len(score))
                prob = np.exp(-np.array(score)/activation)
                res = []
                for i in xrange(n):
                    z = np.sum(prob)
                    r = np.argmax(np.random.multinomial(1, prob/z, 1))
                    res.append(r)
                    prob[r] = 0
                return res
            
            
            deadSamples = []
            deadScores = []
            
            liveSamples = [list(start)]
            liveScores = [0]
            
            while liveSamples:
                probs = predict(liveSamples, empty=empty)
                candidateScore = np.array(liveScores)[:,None] - np.log(probs)
                candidateScore[:, empty] = 1e20
                
                if not oov and outV is not None:
                    candidateScore[:, outV] = 1e20
                
                if avoid:
                    for a in avoid:
                        for i, s in enumerate(liveSamples):
                            n = len(s) - len(start)
                            if n < len(a):
                                candidateScore[i,a[n]] += avoidScore
                                
                liveScores = list(candidateScore.flatten())
                
                
                scores = deadScores + liveScores
                ranks = sample(scores, k)
                n = len(deadScores)
                deadScores = [deadScores[r] for r in ranks if r < n]
                deadSamples = [deadSamples[r] for r in ranks if r < n]
                
                liveScores = [liveScores[r-n] for r in ranks if r >= n]
                liveSamples = [liveSamples[(r-n)//vocabularySize]+[(r-n)%vocabularySize] for r in ranks if r >= n]
                
                def isZombie(s):
                    return s[-1] == eos or len(s) > maxSample
                
                deadScores += [c for s, c in zip(liveSamples, liveScores) if isZombie(s)]
                deadSamples += [s for s in liveSamples if isZombie(s)]
                
                
                liveScores = [c for s, c in zip(liveSamples, liveScores) if not isZombie(s)]
                liveSamples = [s for s in liveSamples if not isZombie(s)]
        
            return deadSamples , deadScores  
        
        
        
        if X is None or isinstance(X,int):
            if X is None:
                i = random.randint(0,len(X_test)-1)
            else:
                i = X
            sys.stdout.flush()
            x = X_test[i]
        else:
            x = [self.word2idx[word.rstrip('^')] for word in X.split()]
            
        if avoid:
            if isinstance(avoid,str) or isinstance(avoid[0], int):
                avoid = [avoid]
            avoid = [a.split() if isinstance(a,str) else a for a in avoid]
            avoid = [outVocabFold([word if isinstance(word,int) else self.word2idx[word] for word in a]) for a in avoid]
        print 'CURRENT HEADLINE:'
        samples = []
        if maxlenDescs == 0:
            skips = [0]
        else:
            skips = range(min(maxlenDescs,len(x)), max(maxlenDescs,len(x)), abs(maxlenDescs - len(x)) // skips + 1)
        for s in skips:
            start = leftPadding(x[:s])
            foldStart = outVocabFold(start)
            sample, score = beamSample(predict=modelPredict, start=foldStart, avoid=avoid, avoidScore=avoidScore, k=k, activation=activation, oov=oov)
            samples += [(s,start,scr) for s,scr in zip(sample,score)]
    
        samples.sort(key=lambda x: x[-1])
        codes = []
        for sample, start, score in samples:
            code = ''
            words = []
            sample = outVocabUnfold(start, sample)[len(start):]
            for word in sample:
                if word == eos:
                    break
                words.append(self.idx2word[word])
                code += chr(word//(256*256)) + chr((word//256)%256) + chr(word%256)
            if short:
                distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
                if distance > -0.6:
                    print score, ' '.join(words)
            else:
                    print score, ' '.join(words)
            codes.append(code)
        return samples
        
        
        
    def train(self):
        maxlenDescs = self.maxlenDescs_training
        maxlenHeads = self.maxlenHeads_predict
        eos = self.eos
        empty = self.empty
        sequenceLen = self.sequenceLen_training
        unknownWords = 10
        vocabularySize = self.vocabularySize
        batchSize = self.batchSize
        outVocabulary = self.vocabularySize-unknownWords
        activationLayer = self.activationLayer
        
        
        def outVocabFold(xStart):
            
            xStart = [x if x < outVocabulary else self.gloveId2Id.get(x,x) for x in xStart]
            outside = sorted([x for x in xStart if x >= outVocabulary])
            outside = dict((x,vocabularySize-1-min(i, unknownWords-1)) for i, x in enumerate(outside))
            xStart = [outside.get(x,x) for x in xStart]
            return xStart
        
        def outVocabUnfold(desc,xStart):
            unfold = {}
            for i, unfold_idx in enumerate(desc):
                fold_idx = xStart[i]
                if fold_idx >= outVocabulary:
                    unfold[fold_idx] = unfold_idx
            return [unfold.get(x,x) for x in xStart]
        
        
        def attentionLayer(X, mask, n=activationLayer, maxlenDescs=maxlenDescs, maxlenHeads=maxlenHeads):
            desc, head = X[:,:maxlenDescs,:], X[:,maxlenDescs:,:]
            head_activations, headWords = head[:,:,:n], head[:,:,n:]
            desc_activations, descWords = desc[:,:,:n], desc[:,:,n:]
            activation = K.batch_dot(head_activations, desc_activations, axes=(2,2))
            activation = activation + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlenDescs],'float32'),1)
            activation = K.reshape(activation,(-1,maxlenDescs))
            activationWeights = K.softmax(activation)
            activationWeights = K.reshape(activationWeights,(-1,maxlenHeads,maxlenDescs))
            descAvgWord = K.batch_dot(activationWeights, descWords, axes=(2,1))
            return K.concatenate((descAvgWord, headWords))
        
        def modelGen():
            model = Sequential()
            model.add(Embedding(self.vocabularySize, self.embeddingDimension, input_length=self.sequenceLen, embeddings_regularizer=None, weights=[self.embeddingMatrix], mask_zero=True,name='embedding_1'))
            model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_1'))
            model.add(Dropout(self.pDense ,name='dropout_1'))
            model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_2'))
            model.add(Dropout(self.pDense ,name='dropout_2'))
            model.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_3'))
            model.add(Dropout(self.pDense ,name='dropout_3'))
            model.add(Lambda(attentionLayer, mask = lambda inputs, mask: mask[:,maxlenDescs:], output_shape = lambda input_shape: (input_shape[0], maxlenHeads, 2*(self.lstmUnits - activationLayer)), name='attentionLayer_1'))
            model.add(TimeDistributed(Dense(self.vocabularySize, name = 'timedistributed_1')))
            model.add(Activation('softmax', name='activation_1'))
            
            return model
        
        def leftPadding(x, maxlenDescs=maxlenDescs, eos=eos):
            if maxlenDescs == 0:
                return [eos]
            l = len(x)
            if l > maxlenDescs:
                x = x[-maxlenDescs:]
                l = maxlenDescs
            return([empty]*(maxlenDescs-l) + x + [eos])
            
            
            
        def teacherForcing(x, flips=None, model=None):
            if flips is None or model is None or flips <= 0:
                return x
            
            batch = len(x)
            assert np.all(x[:,maxlenDescs] == eos)
            probs = model.predict(x, verbose=0, batch_size=batch)
            x_out = x.copy()
            for b in range(batch):
                flips = sorted(random.sample(xrange(maxlenDescs+1,sequenceLen), flips)),
                for input_idx in flips:
                    if x[b,input_idx] == empty or x[b,input_idx] == eos:
                        continue
                    label_idx = input_idx - (maxlenDescs+1)
                    prob = probs[b, label_idx]
                    word = prob.argmax()
                    if word == empty:  
                        word = outVocabulary
                    x_out[b,input_idx] = word
            return x_out
        
        def convSeqLabels(xDescStart, xHeadStart, flips=None, model=None):
            batch = len(xHeadStart)
            assert len(xDescStart) == batch
            x = [outVocabFold(leftPadding(xDesc)+xHead) for xDesc,xHead in zip(xDescStart,xHeadStart)]  
            x = sequence.pad_sequences(x, maxlen=sequenceLen, value=empty, padding='post', truncating='post')
            x = teacherForcing(x, flips=flips, model=model)
            
            y = np.zeros((batch, maxlenHeads, vocabularySize))
            for i, xHead in enumerate(xHeadStart):
                xHead = outVocabFold(xHead) + [eos] + [empty]*maxlenHeads  
                xHead = xHead[:maxlenHeads]
                y[i,:,:] = np_utils.to_categorical(xHead, vocabularySize)
                
            return x, y
        
        def gen(Xdesc, Xhead, batchSize=batchSize, nBatches=None, flips=None, model=None, seed=seed):
            c = nBatches if nBatches else 0
            while True:
                xDescStart = []
                xHeadStart = []
                if nBatches and c >= nBatches:
                    c = 0
                newSeed = random.randint(0, sys.maxint)
                random.seed(c+123456789+seed)
                for b in range(batchSize):
                    t = random.randint(0,len(Xdesc)-1)
        
                    xdesc = Xdesc[t]
                    s = random.randint(min(maxlenDescs,len(xdesc)), max(maxlenDescs,len(xdesc)))
                    xDescStart.append(xdesc[:s])
                    
                    xhead = Xhead[t]
                    s = random.randint(min(maxlenHeads,len(xhead)), max(maxlenHeads,len(xhead)))
                    xHeadStart.append(xhead[:s])
        
               
                c+= 1
                random.seed(newSeed)
        
                yield convSeqLabels(xDescStart, xHeadStart, flips=flips, model=model)
        
        
        
        
        for i in range(unknownWords):
            self.idx2word[self.vocabularySize-1-i] = '<%d>'%i
        
        
        
        for i in range(outVocabulary,len(self.idx2word)):
            self.idx2word[i] = self.idx2word[i] + '^'
        
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = self.validSample, random_state = seed)
        
        self.idx2word[self.empty] = '_'
        self.idx2word[self.eos] = '~'
        
        model = modelGen()
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        
        K.set_value(model.optimizer.lr, np.float32(self.LR))
        
        history = {}
        
        trainGen = gen(X_train, Y_train, batchSize=batchSize, flips=self.flips, model=model)
        valGen = gen(X_test, Y_test, nBatches=self.validSample//batchSize, batchSize=batchSize)
                
        for iteration in range(500):
            print 'Iteration', iteration
            h = model.fit_generator(trainGen, steps_per_epoch=self.trainSample//batchSize, epochs=1, validation_data=valGen, validation_steps=self.validSample)
            for k,v in h.history.iteritems():
                history[k] = history.get(k,[]) + v
            with open(self.historyFile,'wb') as fp:
                pickle.dump(history,fp,-1)
            model.save_weights(self.trainWeightsFile, overwrite=True)
            
    def predict(self, X):
        
        maxlenDescs = self.maxlenDescs_predict
        maxlenHeads = self.maxlenHeads_predict
        activationLayer = self.activationLayer
        batchSize = self.batchSize
        sequenceLen = self.sequenceLen_predict
        
        def attentionLayer(X, mask, n=activationLayer, maxlenDescs=maxlenDescs, maxlenHeads=maxlenHeads):
                desc, head = X[:,:maxlenDescs,:], X[:,maxlenDescs:,:]
                head_activations, headWords = head[:,:,:n], head[:,:,n:]
                desc_activations, descWords = desc[:,:,:n], desc[:,:,n:]
                activation = K.batch_dot(head_activations, desc_activations, axes=(2,2))
                activation = activation + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlenDescs],'float32'),1)
                activation = K.reshape(activation,(-1,maxlenDescs))
                activationWeights = K.softmax(activation)
                activationWeights = K.reshape(activationWeights,(-1,maxlenHeads,maxlenDescs))
                descAvgWord = K.batch_dot(activationWeights, descWords, axes=(2,1))
                return K.concatenate((descAvgWord, headWords))
        
        def modelGen():
            
            
            lstmModel = Sequential()
            lstmModel.add(Embedding(self.vocabularySize, self.embeddingDimension, input_length=sequenceLen, embeddings_regularizer=None, weights=[self.embeddingMatrix], mask_zero=True,name='embedding_1'))
            lstmModel.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_1'))
            lstmModel.add(Dropout(self.pDense ,name='dropout_1'))
            lstmModel.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_2'))
            lstmModel.add(Dropout(self.pDense ,name='dropout_2'))
            lstmModel.add(LSTM(self.lstmUnits, return_sequences=True, dropout = self.pW, recurrent_dropout=self.pU, name='lstm_3'))
            lstmModel.add(Dropout(self.pDense ,name='dropout_3'))
            
            lstmModel.load_weights(self.trainWeightsFile, by_name = True)
            
            model = Sequential()
            model.add(lstmModel)
            model.add(Lambda(attentionLayer, mask = lambda inputs, mask: mask[:,maxlenDescs:], output_shape = lambda input_shape: (input_shape[0], maxlenHeads, 2*(self.lstmUnits - activationLayer)), name='attentionLayer_1'))
    
            return model
        
        if 'embeddingMatrix' not in dir(HeadlineGen):
            with open(self.embeddingFile, 'rb') as fp:
                self.embeddingMatrix, self.idx2word, self.word2idx, self.gloveId2Id = pickle.load(fp)
                
        
        model = modelGen()
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        samples = self.genSamples(model=model, X = X, skips = 2, batchSize = batchSize, k = 10, activation = 1, oov = True, short = False)
        headline = samples[0][0][len(samples[0][1]):]
        headlineText = ' '.join(self.idx2word[w] for w in headline)
        
        return headlineText


model = HeadlineGen()        
#Vocabulary and embedding generation

#model.vocabularyGen()

#training
#model.train()

#prediction
#X = "President Barack Obama 's re-election campaign is fundraising off of comments on Obama 's birth certificate by Mitt Romney 's son Matt ."
#Y = model.predict(X)
#print(Y)

model = HeadlineGen()
model.vocabularyGen()
           
