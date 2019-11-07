#################################################################
# The program is made for HackerEarth Submission by Devendra
#all the training and test files should be in the folder where SolutionCode.py is running
#################################################################
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
########################################################################
NFeat=3600     # NFeat controls the number of elements in feature vector
########################################################################
data= pd.read_csv(r"hm_train.csv")     # loading training data file
nonulldata= data.dropna(axis=0)                                                           # droping any row with invalid entry
ROI_in_data= nonulldata.drop(["num_sentence","hmid","reflection_period"],axis=1);         # taking out region of interest from data
ROI_in_data["cleaned_hm"]= ROI_in_data["cleaned_hm"].map(lambda x : word_tokenize(str.lower(x)))     # converting statements in lowercase then taking out all the words
features= ROI_in_data.drop(["predicted_category"],axis=1);                                # taking out features (vectors of words)
emotion= ROI_in_data.drop("cleaned_hm",axis=1)                                            # taking out labels for features
bag_with_emotion= []                                                                      # this will have features and lable as tuple(list(feature),label)
bag=[]                                                                                    # this will have all the words in the given traing dataset

######### the coming loop converts features and labels in the form of list of tuple(list(feature),label) ##########3
for (words,emotion) in zip(ROI_in_data["cleaned_hm"],ROI_in_data["predicted_category"]):
    bag_with_emotion.append((list(words),emotion))
    for word in words:
        bag.append(word)
########################################################################
lemmatizer = WordNetLemmatizer()
stop_words=set(stopwords.words("english"))                                                # produces most common words used in english
Important_bag=[]                                                                          # this will have words that are not very common in english language
for word in bag:
    if word not in stop_words:
        Important_bag.append(lemmatizer.lemmatize(word))



bag= nltk.FreqDist(Important_bag)                                                          # bag contains words as {word: frequency} in decreasing order of their frequescy
imp_words= list(bag.keys())[:NFeat]                                                        # imp_words will have NFeat number of most important words now


##### the comming function transforms the statementWordsVector which contain words into a Nfeatx1 row of binary elements.#####
def transformerToVector(statementWordsVector):
    for i in range(len(statementWordsVector)):
        statementWordsVector[i]= lemmatizer.lemmatize(statementWordsVector[i]);
    WordsVector=[]
    for i in range(len(imp_words)):
        if imp_words[i] in statementWordsVector:
            WordsVector.append(1)
        else:
            WordsVector.append(0)
    return WordsVector
##############################################################################################################################
# transformToVector is the function which actually produces the feature vector which wwill be used by us in training
##############################################################################################################################
feature_bag=[]                                          # feature_bag will contain tuples(Vectorised_feature_in_terms_of_01,label)
for emotion_bag in bag_with_emotion:
    feature_bag.append((list(transformerToVector(emotion_bag[0])),emotion_bag[1]))
featureMat=[]                                           # this will have all the vectorised features as rows
label=[]                                                # this will have all the labels
for i in feature_bag:
    featureMat.append(i[0])
    label.append(i[1])

train_features, test_features, train_res, test_res= train_test_split(featureMat,label)     #splitting the data in train and test to validate

###################################################################################################################################
##########################Pre processing of data is completed######################################################################
###################################################################################################################################

#####Child Models will be trained now################################
#####################################################################
cls= MultinomialNB()
cls.fit(train_features,train_res)
res_model_MNB=cls.predict(test_features)
print("MultinomialNB\n############################################################################")
print(classification_report(test_res,res_model_MNB))
#####################################################################
cls1= BernoulliNB()
cls1.fit(train_features,train_res)
res_model_BNB=cls1.predict(test_features)
print("BernoulliNB\n############################################################################")
print(classification_report(test_res,res_model_BNB))
#####################################################################
cls2= LogisticRegression()
cls2.fit(train_features,train_res)
res_model_LR=cls2.predict(test_features)
print("LogisticRegression\n############################################################################")
print(classification_report(test_res,res_model_LR))
####################################################################
cls3= SGDClassifier(loss="log")
cls3.fit(train_features,train_res)
res_model_SGDC=cls3.predict(test_features)
print("SGDClassifier\n############################################################################")
print(classification_report(test_res,res_model_SGDC))
#####################################################################
cls4= LinearSVC()
cls4.fit(train_features,train_res)
res_model_LSVC=cls4.predict(test_features)
print("LinearSVC\n############################################################################")
print(classification_report(test_res,res_model_LSVC))
######################################################################
#########################Training over################################

##########################pre-processing of test data starts here##############################
########this is ver similar to the preprocessing of traing data################################

dataNew= pd.read_csv(r"hm_test.csv")
nonulldataNew= dataNew.dropna(axis=0)
ROI_in_dataNew= nonulldataNew.drop(["num_sentence","reflection_period"],axis=1);
ROI_in_dataNew["cleaned_hm"]= ROI_in_dataNew["cleaned_hm"].map(lambda x : word_tokenize(str.lower(x)))
features= ROI_in_dataNew.drop(["hmid"],axis=1);
emotion= ROI_in_dataNew.drop("cleaned_hm",axis=1)
bag_with_emotion= []
bag=[]
for (words,emotion) in zip(ROI_in_dataNew["cleaned_hm"],ROI_in_dataNew["hmid"]):
    bag_with_emotion.append((list(words),emotion))
    for word in words:
        bag.append(word)
feature_bag=[]
for emotion_bag in bag_with_emotion:
    feature_bag.append((list(transformerToVector(emotion_bag[0])),emotion_bag[1]))
featureMat=[]
label=[]
for i in feature_bag:
    featureMat.append(i[0])
    label.append(i[1])
################################Pre processing ends here#########################################
classifierLbl={0:'achievement', 1:'affection', 2:'bonding', 3:'enjoy_the_moment', 4:'exercise', 5:'leisure',6:'nature'}  #classifier labels mapped to index for FATHER_Classifier
##################################################################################################

################################Father_Classifier#################################################
def FATHER_Classifier(dataNew):
    vec2=[]
    vec2.append(dataNew)
    votes={'achievement': 0, 'affection':0, 'bonding':0, 'enjoy_the_moment':0, 'exercise': 0, 'nature': 0,'leisure':0}

    ###voter is a function for 'hard voiting'. this is used in debugging with tweeking for accuracy and reliability####
    def voter(p):
       if p=="bonding":
           votes["bonding"]= votes["bonding"]+1
       elif p == "achievement":
           votes["achievement"] = votes["achievement"] + 1
       elif p == "leisure":
           votes["leisure"] = votes["leisure"] + 1
       elif p == "enjoy_the_moment":
           votes["enjoy_the_moment"] = votes["enjoy_the_moment"] + 1
       elif p == "nature":
           votes["nature"] = votes["nature"] + 1
       elif p == "exercise":
           votes["exercise"] = votes["exercise"] + 1
       elif p == "affection":
           votes["affection"] = votes["affection"] + 1
       else:
           pass

    #####################Child classifiers are making prdictions here#####################
    ####we can add as many as we desire after training them above in training section#####

    p=cls.predict(vec2)
    prob = cls.predict_proba(vec2)
    #print("prediction probability: ", prob, "  Prediction: ", p);
    voter(p)
    #print("+++++++++++++++++++++++++++++++ Child Prediction +++++++++++++++++++++++++++++")
    p = cls1.predict(vec2)
    prob1 = cls1.predict_proba(vec2)
    #print("prediction probability: ", prob1, "  Prediction: ", p);
    voter(p)

    p = cls2.predict(vec2)
    prob2 = cls2._predict_proba_lr(vec2)
    #print("prediction probability: ", prob2, "  Prediction: ", p);
    voter(p)

    p = cls3.predict(vec2)
    prob3 = cls3.predict_proba(vec2)
    #print("prediction probability: ", prob3, "  Prediction: ", p);
    voter(p)

    p= cls4.predict(vec2)
    prob4 = cls4._predict_proba_lr(vec2)
    #print("prediction probability: ",prob4,"  Prediction: ", p );
    voter(p)
    ######################## Prediction for childs is over ####################################
    ########################## finding class with largest chance to be correct ################
    probabilty= prob+prob1+prob2+prob3+prob4
    indexi=np.argmax(probabilty[0])
   # print("=====================================================================\nprobability final Score: ", probabilty,"Class to be predicted: ", probabilty.max(),"\n=====================================================================\n")
    #print("+++++++++++++++++++++++++++++ Child Prediction Over +++++++++++++++++++++++++++")
    return (probabilty.max())/5, classifierLbl[indexi]
################################# FATHER_Classifier ends here #################################


######################### Prediction for the test data will be done here ######################
result= []
hmidres=[]
for (testinst,lbl) in zip(featureMat,label):
    probability_of_prediction, prediction=  FATHER_Classifier(testinst)
    print("Final Predicted as:      ",prediction,"\nWith probability(highest among other classes): ",probability_of_prediction,"\n=====================================================================")
    result.append(prediction)
    hmidres.append(lbl)
########################### Prediction/classification is over #################################
out= pd.DataFrame(list(zip(hmidres,result)),columns=["hmid","predicted_category"])      # coverting the results into dataframe
out.to_csv(path_or_buf="submission.csv",index=False)                                    # writing dataframe to a submission.csv file

########################################   Program Ends Here ##################################