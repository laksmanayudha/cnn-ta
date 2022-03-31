import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def select_features_rasio(data, vocab_indexer, chi_results, rasio):
    sorted_words = chi_results.sort_values(by=['Chi Square'], ascending=False)['word list']
    index = round(len(sorted_words)*rasio)
    selected_features = list(sorted_words[:index])
    
    selected_words = []
    for index, tfidf in enumerate(data):
        print("data", index+1)
        selected = []
        for feature in selected_features:
          selected.append(tfidf[vocab_indexer[feature]])
        selected_words.append([selected])
    return selected_words

def select_features_nWords(data, vocab_indexer, chi_results, n):
    sorted_words = chi_results.sort_values(by=['Chi Square'], ascending=False)['word list']
    # index = round(len(sorted_words)*rasio)
    selected_features = list(sorted_words[:n])
    
    selected_words = []
    for index, tfidf in enumerate(data):
        print("data", index+1)
        selected = []
        for feature in selected_features:
          selected.append(tfidf[vocab_indexer[feature]])
        selected_words.append([selected])
    return selected_words

def select_nWords(chi_results, n):
    sorted_words = chi_results.sort_values(by=['Chi Square'], ascending=False)['word list']
    # index = round(len(sorted_words)*rasio)
    selected_features = list(sorted_words[:n])
    return selected_features

def select_features(data, chi_results, rasio):
    sorted_words = chi_results.sort_values(by=['Chi Square'], ascending=False)['word list']
    index = round(len(sorted_words)*rasio)
    selected_features = list(sorted_words[:index])
    
    selected_words = []
    for index, text in enumerate(data):
        print("data", index+1)
        selected_words.append(" ".join(list(filter(lambda x: x in selected_features, text))))
    
    return selected_words

def select_features_oldVersion(data, chi_results, rasio):
    sorted_words = chi_results.sort_values(by=['Chi Square'], ascending=False)['word list']
    index = round(len(sorted_words)*rasio)
    selected_features = list(sorted_words[:index])
    
    selected_words = []
    for index, text in enumerate(data):
        print("data", index+1)
        selected_words.append(list(filter(lambda x: x in selected_features, text)))
    
    return selected_words

def get_term_binary_matrix(input_doc_list):
    #unique word and word count
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(input_doc_list)
    word_list = vectorizer.get_feature_names()

    #binary word document matrix
    # vectorizer = CountVectorizer(binary=True)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(input_doc_list)
    word_binary_matrix = X.toarray()
    count_list = word_binary_matrix.sum(axis=0)
    ##return
    return word_list,count_list,word_binary_matrix

def get_ABCD(word_binary_matrix,label_array):

        A=[]
        B=[]
        C=[]
        D=[]
        for i in range(word_binary_matrix.shape[1]):
            computed_result=Counter(label_array * 2 + word_binary_matrix[:,i])
            A.append(computed_result[1])
            B.append(computed_result[3])
            C.append(computed_result[0])
            D.append(computed_result[2])

        A=np.array(A)
        B=np.array(B)
        C=np.array(C)
        D=np.array(D)
        N=A+B+C+D
        return A,B,C,D,N

def get_binary_label(label_array):
    #get numpy array
    label_array=np.array(label_array)
    unique_label=np.unique(label_array)
    #if not binary coded already, do so
    if 0 in unique_label and 1 in unique_label:
        pass
    else:
        label_array=np.where(label_array==unique_label[0],1,0)
    return label_array

def ChiSquare(A,B,C,D,N):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (N*((A*D)-(C*B))**2)/((A+B)*(A+C)*(B+D)*(C+D))
    
def get_score(target, input_doc_list):
    #labels as numpy array
    numpy_target=np.array(target)

    #get word, count, binary matrix
    word_list,count_list,word_binary_matrix=get_term_binary_matrix(input_doc_list)
    
    result_dict={}

    #for each class
    for calc_base_label in list(set(target)):
        #get binary labels
        label_array=np.where(numpy_target==calc_base_label,1,0)

        #get ABCDN
        B,A,D,C,N=get_ABCD(word_binary_matrix,label_array)

        #create DF
        out_df=pd.DataFrame({'word list':word_list,'word occurence count':count_list})

        out_df['Chi Square']=ChiSquare(A,B,C,D,N)

        ##assign to dict for master calculation
        result_dict[calc_base_label]=out_df
        
    final_results_chi=pd.DataFrame()
        
    #final result
    final_results=pd.DataFrame({'word list':out_df['word list'],'word occurence count':out_df['word occurence count']})
        
    for calc_base_label in list(set(target)):
        label_df_chi=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'CHI_'+str(calc_base_label):result_dict[calc_base_label]['Chi Square']})
        if final_results_chi.shape[0]:
            final_results_chi=final_results_chi.merge(label_df_chi,on=['word list'])
        else:
            final_results_chi=label_df_chi

        ##final calculation
        if calc_base_label==list(set(target))[-1]:
            label_df_chi=pd.DataFrame({'word list':final_results_chi['word list'],
                                      'Chi Square':final_results_chi.max(axis=1)})
            #assign to final result df
            if final_results.shape[0]:
                final_results=final_results.merge(label_df_chi,on=['word list'])
            else:
                final_results=label_df_chi

    return final_results