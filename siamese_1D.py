import numpy.random as rng
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score

import os,json

from sys import stdout
def flush(string):
    stdout.write('\r')
    stdout.write(str(string))
    stdout.flush()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,X_train,y_train,X_val,y_val):
        self.data = {'train':X_train,'val':X_val}
        self.labels = {'train':y_train,'val':y_val}
        train_classes = list(set(y_train))
        np.random.seed(10)
#         train_classes = sorted(rng.choice(train_classes,size=(int(len(train_classes)*0.8),),replace=False) )
        self.classes = {'train':sorted(train_classes),'val':sorted(list(set(y_val)))}
        self.indices = {'train':[np.where(y_train == i)[0] for i in self.classes['train']],
                        'val':[np.where(y_val == i)[0] for i in self.classes['val']]
                       }
        print(self.classes)
        print(len(X_train),len(X_val))
        print([len(c) for c in self.indices['train']],[len(c) for c in self.indices['val']])
        
    def set_val(self,X_val,y_val):
        self.data['val'] = X_val
        self.labels['val'] = y_val
        self.classes['val'] =  sorted(list(set(y_val)))
        self.indices['val'] =  [np.where(y_val == i)[0] for i in self.classes['val']]
    
    def set_train(self,X,y):
        self.data['train'] = X
        self.labels['train'] = y
        self.classes['train'] =  sorted(list(set(y)))
        self.indices['train'] =  [np.where(y == i)[0] for i in self.classes['train']]
    

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w= X.shape

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=True)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, w)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        np.random.shuffle(targets)
        for i in range(batch_size):
            category = categories[i]
            n_examples = len(X_indices[category])
            if(n_examples==0):
                print("error:n_examples==0",n_examples)
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:] = X[X_indices[category][idx_1]]
            #pick images of same class for 1st half, different for 2nd
            if targets[i]==1:
                category_2 = category  
                idx_2 = (idx_1 + rng.randint(1,n_examples)) % n_examples
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
                n_examples = len(X_indices[category_2])
                idx_2 = rng.randint(0, n_examples)
            pairs[1][i,:] = X[X_indices[category_2][idx_2]]
        return pairs, targets, categories
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    


    def make_oneshot_task2(self,idx,s="val",support_set=[]):
        """Create pairs_list of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        X_labels = self.labels[s]
        
        X_train=self.data['train']
        indices_train = self.indices['train']
        classes_train = self.classes['train']
        N = len(indices_train)
        
        _, w = X.shape
        
        test_image = np.asarray([X[idx]]*N)

        if(len(support_set) == 0):
            support_set = np.zeros((N,w))
            for index in range(N):
                support_set[index,:] = X_train[rng.choice(indices_train[index],size=(1,),replace=False)]
            support_set = support_set

        targets = np.zeros((N,))
        true_index = classes_train.index(X_labels[X_labels.index[idx]])
        targets[true_index] = 1
        
#         targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, classes_train)
        categories = classes_train
     
        pairs = [test_image,support_set]
        
        return pairs, targets,categories
            
    def test_oneshot2(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        k = len(self.labels[s])
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        probs_all = []
        err_print_num = 0
        support_set=[]
        for idx in range(k):
        
            inputs, targets,categories = self.make_oneshot_task2(idx,s,support_set)
            support_set=inputs[1]
            n_classes, w = inputs[0].shape
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
            elif verbose and err_print_num<1:
                err_print_num = err_print_num +1
                print(targets)
#                 print(categories)
                print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
            preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
            probs_all.append(probs)
#             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct,np.array(preds),np.array(probs_all)
    
    def test_fewshot2(self,model,N,k,s="val",verbose=0,shots=1):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        k = len(self.labels[s])
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        probs_all = []
        err_print_num = 0
        X_train=self.data['train']
        X=self.data[s]
        indices_train = self.indices['train']
        N = len(indices_train) 
        _, w = X.shape
       
        all_support_set=np.zeros((shots,N,w))
        all_selected_idx=[]
        for index in range(N):
            selected_idx=rng.choice(indices_train[index],size=(shots,),replace=False)
            all_selected_idx.append(selected_idx)
            selected= X_train[rng.choice(indices_train[index],size=(shots,),replace=False)]
            for shot in range(shots):
                all_support_set[shot,index,:]=selected[shot]
                
        preds_few_shot = []
        prods_few_shot = []
        scores = []
        for shot in range(shots):
            n_correct = 0
            preds = []
            probs_all = []
            for idx in range(k):
                inputs, targets,categories = self.make_oneshot_task2(idx,s,all_support_set[shot])
                n_classes, w = inputs[0].shape
                probs = model.predict(inputs)
                if np.argmax(probs) == np.argmax(targets):
                    n_correct+=1
                elif verbose and err_print_num<1:
                    err_print_num = err_print_num +1
                    print(targets)
        #                 print(categories)
                    print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                    plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
                preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                probs_all.append(probs)
        #             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
            percent_correct = (100.0*n_correct / k)
            percent_correct,preds,probs_all=percent_correct,np.array(preds),np.array(probs_all)
            if verbose:
                print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
            print(percent_correct,preds.shape,probs_all.shape)
            scores.append(percent_correct)
            preds_few_shot.append(preds[:,1])
            prods_few_shot.append(probs_all)
            preds = []
            for line in np.array(preds_few_shot).T:
                pass
                preds.append(np.argmax(np.bincount(line)))
        #             utils.confusion_plot(np.array(preds),data.y_test) 
            prod_preds = np.argmax(np.sum(prods_few_shot,axis=0),axis=1).reshape(-1)

            score_few_shot = accuracy_score(self.labels['val'],np.array(preds))*100

            score_few_shot_prob = accuracy_score(self.labels['val'],prod_preds)*100
        print('{}_shot Accuracy based on {} one-shot prediction:'.format(shots,shots),score_few_shot)  
        print('{}_shot Accuracy based on _probabilty:'.format(shots),score_few_shot_prob)
        return score_few_shot,score_few_shot_prob,preds,prod_preds

          
def train_and_test_oneshot(settings,siamese_net,siamese_loader):
    settings['best'] = -1 
    settings['n'] = 0
    print(settings)

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print('*'*40+"training"+ '*'*40)
    
    #Training loop
    for i in range(settings['n'], settings['n_iter']):
        (inputs,targets,_)=siamese_loader.get_batch(settings['batch_size'])
        n_classes, w = inputs[0].shape

        loss=siamese_net.train_on_batch(inputs,targets)

        if i == 0:
            k = len(siamese_loader.labels['val'])
            N=settings['N_way']
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
            
        if i % settings['evaluate_every'] == 0:
            val_acc,preds,probs_all = siamese_loader.test_oneshot2(siamese_net,settings['N_way'],settings['n_val'],verbose=0)
            preds = np.array(preds)
            if val_acc >= settings['best'] :
                print("\niteration {} evaluating: {}%".format(i,val_acc))
#                 print(loader.classes)
#                 score(preds[:,1],preds[:,0])
#                 print("\nsaving")
                siamese_net.save(weights_path)
                settings['best'] = val_acc
                settings['n'] = i
                with open(os.path.join(weights_path+".json"), 'w') as f:
                    f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': ')))

        if i % settings['loss_every'] == 0:
            flush("interation {}, Loss: {:.5f},".format(i,loss))
            
    return settings['best']