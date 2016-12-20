import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, log_loss, average_precision_score, precision_score, recall_score, f1_score)
from sklearn.metrics import (precision_recall_fscore_support, precision_recall_curve)
import time
from sklearn import (metrics, cross_validation, linear_model)
from sklearn.externals import joblib
from sklearn.datasets import load_boston
import os
import gzip
import pickle
import pandasql as ps
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sets import Set
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import (median_absolute_error, r2_score)


"""
Generating lift charts and saving in savePath
"""
def lift(sortby, pred, actual, weight=None, savePath="e:\\mydata\\",n=10, plot=True, std=None, title="Gini", show=True):
    """
    sort, pred, actual, weight can be a 1-d numpy array or a single column of a pandas dataframe
    sortby = sort by variable
    pred = predicted values
    actual = actual values
    weight = weights (leave blank if NA)
    n = number of buckets
    plot = True if plot the resulting lift chart
    std = if not None, shows the error bar for each bucket
    """
    if weight is None:
        weight = np.ones_like(pred)

    def weighted_std(values, weight):
        """
        weighted std of an array
        this is slightly biased but shouldn't matter for big n
        """
        m = np.average(values, weights=weight)
        return np.sqrt(np.average((values-m)**2, weights=weight))

    r = np.vstack((sortby, pred, actual, weight)).T
    r = r[sortby.argsort()].T
    cumm_w = np.cumsum(r[3])
    cumm_y = np.cumsum(r[2]*r[3])
    total_w = np.sum(weight)
    gini = 1-2*(np.sum(cumm_y*r[3])/(np.sum(r[2]*r[3])*total_w))
    idx = np.clip(np.round(cumm_w*n/total_w + 0.5), 1, n) - 1
    lift_chart = np.zeros((n,7))
    for i in range(n):
        lift_chart[i][0] = np.sum(r[3][idx==i]) #num observations in each bucket
        lift_chart[i][1] = np.sum(r[1][idx==i]*r[3][idx==i])/lift_chart[i][0] #mean prediction
        lift_chart[i][2] = np.sum(r[2][idx==i]*r[3][idx==i])/lift_chart[i][0] #mean actual
        lift_chart[i][3] = weighted_std(r[1][idx==i],r[3][idx==i]) #weighted std
        lift_chart[i][4] = np.average(r[0][idx==i],weights=r[3][idx==i])#mean sortby variable
        lift_chart[i][5] = np.min(r[0][idx==i]) #min sortby variable
        lift_chart[i][6] = np.max(r[0][idx==i])#max sortby variable
    if plot==True:
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        x = range(1,n+1)
        ax.plot(x, lift_chart[:,1], "b", label="Predicted")
        ax.plot(x,  lift_chart[:,2], "r", label="Actual")
        ax.grid(True)
        ax.set_xlabel("Buckets (Equal Exposure)")
        ax.set_ylabel("Target")
        ax.legend(loc=2)
        if std is not None:
            ax.fill_between(x, lift_chart[:,1]+std*lift_chart[:,3],lift_chart[:,1]-std*lift_chart[:,3],
                            color="b", alpha=0.1)
        ax.set_title(title + "\n Gini: " + format(gini, ".4f"))
        fig.savefig(savePath + title + " Lift Chart.png")
        if(show==True):
            plt.show()
        plt.close(fig)
    return gini, lift_chart

"""
Generates a generic lift chart (without Gini)
"""
def genericlift(pred, actual, filename, savePath="e:\\mydata\\",plot=True, show=True):
    pColor="b"
    aColor="r"
    n = len(pred)
    if plot==True:
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(range(1, n+1), pred, pColor, label="Predicted")
        ax.plot(range(1, n+1),  actual, aColor, label="Actual")
    
        ax.grid(True)
        #ax.set_xlabel("Gini: %.12f"  %gini)    
        ax.set_xlabel("Lift chart")    
        ax.legend(loc=2)
        #plt.rcParams.update({'figure.figsize': (3,3)})
        #DefaultSize = fig.get_size_inches()
        fig.set_size_inches( (DefaultSize[0]*1.5, DefaultSize[1]*1.5) )
        fig.savefig(savePath + filename + ".png")
        plt.close(fig)
            
"""
Encoding features so they could be used in GBM/XGBoost
"""
def getFeatures(value, featureList, idx):
    features = {}
    if idx%10000==0: print idx
    for f in featureList:
        features[f] = value[f][idx]
    return features

"""
Determine, for a list of variables, the top (max) values for every variable listed
"""
def getStats(values, variables, dataframename, max):
    valuesParameters = {}
    
    for v in variables:        
        tmp = Set()
        res = ps.sqldf("select count(*), " + str(v) + " from " + dataframename + " group by " + str(v) + " order by count(*) desc limit " + str(max) + ";",values) #locals())
        for item in res[v]:
            tmp.add(item)
        
        valuesParameters[v] = tmp

    return valuesParameters
    
"""
Encoding features so they could be used in GBM/XGBoost. Only take the X values for every variables, put others in 'Other - not in top values'
"""
def smartGetFeatures(value, featureList, featureListCategorical, varsToHandle, idx):
    features = {}
    if idx%10000==0: print idx
    for f in featureList:
        features[f] = value[f][idx]
    
    for f in featureListCategorical:
        features[f] = returnOther(varsToHandle, f, value[f][idx])
    return features

"""
Determine, based on varsToHandle, whether a variable is in the top values or not
"""
def returnOther(varsToHandle, varName, value):
    if str(value) == 'nan' or value == None or str(value) == 'None':
        return value
        
    canuse = varsToHandle[varName]
    if value in canuse:
        return value
    else:
        return "Other - not in top values"
        
"""
calculate the Gini score
"""
def mygini(sortby, pred, actual, weight=None, n=10, plot=True, std=None):
    if weight is None:
        weight = np.ones_like(pred)

    def weighted_std(values, weight):

        m = np.average(values, weights=weight)
        return np.sqrt(np.average((values-m)**2, weights=weight))

    r = np.vstack((sortby, pred, actual, weight)).T
    r = r[sortby.argsort()].T
    cumm_y = np.cumsum(r[2]*r[3])
    total_w = np.sum(weight)
    gini = 1-2*(np.sum(cumm_y*r[3])/(np.sum(r[2]*r[3])*total_w))
    return gini

"""
generate feature importance list for GBM
"""
def returnFeatureImportance(gbm_model, vec, file):
    feature_importance = gbm_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    featuresNames = []
    featureImps =[]
    for item in sorted_idx[::-1][:]:
        featuresNames.append(np.asarray(vec.feature_names_)[item])
        featureImps.append(feature_importance[item])
    featureImportance = pd.DataFrame([featuresNames, featureImps]).transpose()
    featureImportance.columns = ['FeatureName', 'Relative Importance']
    featureImportance.to_csv(file)
    #featureImportance.to_csv(path + "FeatureImportance.csv")
    return featureImportance
"""
generate feature importances list for GBM with pandas df
"""
def returnFeatureImportanceDF(gbm_model, df, file):
    feature_importance = gbm_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    featuresNames = []
    featureImps =[]
    for item in sorted_idx[::-1][:]:
        featuresNames.append(df.columns.values[item])
        featureImps.append(feature_importance[item])
    featureImportance = pd.DataFrame([featuresNames, featureImps]).transpose()
    featureImportance.columns = ['FeatureName', 'Relative Importance']
    featureImportance.to_csv(file)
    # featureImportance.to_csv(path + "FeatureImportance.csv")
    return featureImportance

"""
generate feature importance list for XGBoost
"""
def printFeatureImportanceXGBoost(vec, pd, bst, outputDir=""):
    ts = pd.Series(bst.get_fscore())
    features = vec.feature_names_
    mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
    ts.index = ts.reset_index()['index'].map(mapFeat)
    feature_importance=ts.order()[::-1]
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    
    for i in feature_importance.keys():
        print str(i) + "\t" + str(feature_importance[i])
       
    if outputDir != "": 
        feature_importance.to_csv(outputDir + "FeatureImportance.csv")

def printFeatureImportanceXGBoostFeatureList(gbdt, outputDir=""):
    fscore = [ (v,k) for k,v in gbdt.get_fscore().iteritems() ]
    fscore.sort(reverse=True)
    #print fscore
    features=[k for (i,k) in fscore]
    scores=[i for (i,k) in fscore]
    k=pd.DataFrame({'Feature':features, 'Importance':scores})
    k['Percent_Importance'] = 100.0 * (k['Importance'] / k['Importance'].max())
    print k
    if outputDir != "":
        k.to_csv(outputDir + "FeatureImportance.csv")

"""
Generate partial dependant plot
"""
def generatePDP(modelObj, featureVector, trainingX, outputFolder,importance=10):
    #Create Partial Dependenct directory to hold all PD plots
    pdDir = outputFolder
    #if the output Partial Dependency Directory doesn't exist, create it
    if not os.path.exists(os.path.dirname(pdDir)):
        print "Output Directory: " + pdDir + " Doesn't exist. Creating it now"
        os.mkdir(os.path.dirname(pdDir))
    # to generate feature importance
    featureImportanceDF = returnFeatureImportance(modelObj, featureVector)
    #Select only the important features
    featureImportanceDF = featureImportanceDF[featureImportanceDF['Relative Importance'] > importance]
    # to generate PDP, create a list of features
    featureId =[]
    featureName = []
    for k, feature in enumerate(featureVector.feature_names_,):
        featureId.append(k)
        featureName.append(feature)
    features = pd.DataFrame([featureId, featureName]).transpose()
    features.columns = ['FeatureId', 'FeatureName']
    #Get the feature id for the important features
    featureImportanceDF = pd.merge(featureImportanceDF, features, how='left', on='FeatureName')

    #Generate PD Plots
    for i in range(featureImportanceDF['FeatureName'].size):
         feature = [featureImportanceDF['FeatureId'][i]]
         featName = featureImportanceDF['FeatureName'][i].replace('/','_')
         fig, axs = plot_partial_dependence(modelObj,trainingX,feature,featureVector.feature_names_,n_jobs=-1)
         plt.subplots_adjust(top = 0.9)
         #axs.set_xlabel(featName)
         #save the plot in the output directory with the feature name as file name
         fig.savefig(pdDir + featName + "_PD.png")
         plt.close(fig)

"""
Convert continuous integer variable to a boolean.
"""
def booleanize (x): return x>=0.5

"""
Store pickle file. Can be a model object or a vec object.
"""
def storePickle(filename, object):
    f = gzip.open(filename,'wb')
    pickle.dump(object,f)
    f.close()
    
"""
Load pickle from file.
"""
def loadPickle(filename):
    f = gzip.open(filename,'rb')
    object = pickle.load(f)
    f.close()
    return object

"""
Implementation of Owen Zhang's leave-one out experience variables
"""

def getNwayS(df, tst, y, scaling=0.025):
    """
    df: training dataframe of categorical vars to be transformed
    tst: testing dataframe of categorical vars to be transformed
    y: target in the training set, input as column of the training dataframe
       for classification y must be 0/1
    scaling: sd of normally generated mult. factor, scaling is initially set to 0.025

    returns: (training transformed column, testing transformed column)
    """
    df['y']=y
    #For a group level,  attach the average y value of all others of the same group
    #For a group level in the test set, attach the average y value in the training set
    z=df.join(df.groupby(list(set(df.columns.values)-{'y'})).agg([np.mean, len]), on=list(set(df.columns.values)-{'y'}),how='left')
    k=tst.join(df.groupby(list(set(df.columns.values)-{'y'})).agg([np.mean, len]), on=list(set(tst.columns.values)), how='left')

    z['t']=(z.ix[:,-1]*z.ix[:,-2]-z['y'])/(z.ix[:,-1]-1)

    #if level is of frequency 1, set its value to a uniformly random value in range of values
    z.ix[z.ix[:,-2]==1,-1]=np.random.uniform(low=min(z.ix[:,-3]),high=max(z.ix[:,-3]),size=len(z.ix[z.ix[:,-2]==1,-1]))

    #if tst has NAs fill them with 0s
    k.ix[pd.isnull(k.ix[:,-1]),-2]=np.zeros(len(k.ix[pd.isnull(k.ix[:,-1]),-2]))

    #apply scaling
    if(scaling>0):
        z['t']=z['t']*np.random.normal(loc=1.0,scale=scaling,size=len(z['t']))
        k.ix[:,-2]=k.ix[:,-2]*np.random.normal(loc=1.0,scale=scaling,size=len(k.ix[:,-2]))

    return (z['t'], k.ix[:,-2])


class getNway(TransformerMixin):
    """
    Transformer class to convert categorical variable to experience variable

    X: dataframe of values to transform (could be multiple columns
    y: target in the training set, input as column of the training dataframe
       for classification y must be 0/1, include in transform for train
    scaling: std of normal distributed scaling factor (0.025)
    """
    def __init__(self, scaling=None):
        self.scaling = scaling

    def fit(self, X, y, **fit_params):
        X['y']=y
        self.grpd_=X.groupby(list(set(X.columns.values)-{'y'})).agg([np.mean, len])
        return self

    def transform(self, X, y=None, **fit_params):

        X=X.join(self.grpd_, how='left', on=list(set(X.columns.values)))
        length=X.ix[:,-1]
        tr_mean=X.ix[:,-2]
        if y is not None:
            tr_mean=(length*tr_mean-y)/np.maximum((length-1).astype(float), np.ones(len(length)))

        #if level is of frequency 1, set its value to a uniformly random value in range of values
        tr_mean[length==1]=-10.0#np.random.uniform(low=min(tr_mean), high=max(tr_mean),size=len(tr_mean[length==1]))

        #if NAs fill them with 0s
        #tr_mean[pd.isnull(tr_mean)]=np.zeros(len(tr_mean[pd.isnull(tr_mean)]))

        if(self.scaling):
            tr_mean=tr_mean*np.random.normal(loc=1.0, scale=self.scaling, size=len(tr_mean))

        return tr_mean


"""
tf-idf function
"""
def tfidf(min_df, ngram_lower, ngram_higher):
    t_pat = r"(?u)\b[A-Za-z$\']+\b"
    tf_trans = TfidfVectorizer(min_df=min_df,
                            max_df=100000,
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            ngram_range=(ngram_lower, ngram_higher),
                            token_pattern=t_pat,
                            #tokenizer=tokenize,
                            decode_error = "ignore",
                            strip_accents='unicode',
                            analyzer='word',
                            stop_words = 'english',
                            norm="l2",
                            lowercase=True)
    return(tf_trans)

"""
Some useful transformer classes
"""

class DataframeColumnExtractor(TransformerMixin):
    """
    Transformer class to extract column(s) from dataframe
    input: column to extract
    """
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class DenseTransformer(TransformerMixin):
    """
    Transformer class to convert sparse dataframe to dense
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()



class DataframeColumnDrop(TransformerMixin):
    """
    Transformer class to drop column
    """

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.column, axis=1)


"""
Function which prints some commom classification metrics
"""

def classification_metrics(actual, predicted, threshold=0):
    """
        actual: 0/1 vector of actual y values
        predicted: 0/1 predicted values or predicted probabilities
        threshold: if predicted represents probabilities set to threshold for positive
    """
    if threshold > 0:
        print "AUC: %f\n" %metrics.roc_auc_score(actual, predicted)
        print "Log loss: %f\n" %log_loss(actual, predicted)
        print "Area under PR-curve (average precision): %f\n" %average_precision_score(actual, predicted)
        create_precision_recall_curve(actual, predicted)
        predicted = predicted >= threshold

    df_confusion = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print "\nConfusion Matrix: \n"
    print df_confusion
    print "\n\nAccuracy: %f\n" %accuracy_score(actual, predicted)

    print "Percentage of True 1's: %.3f" %(100*len(actual[np.where(actual==1)])/float(len(actual))),"%"
    print "\nPrecision: ", precision_recall_fscore_support(actual, predicted)[0]
    print "\nRecall: ", precision_recall_fscore_support(actual, predicted)[1]
    print "\nF1-score: ", precision_recall_fscore_support(actual, predicted)[2]

"""
Function which prints some regression metrics
"""

def regression_metrics(actual, predicted):
    """
    actual: array of actual values
    predicted: array of predicted real values
    """
    print 'Gini:', mygini(predicted, predicted, actual)
    print 'RMSE: ', np.sqrt(mean_squared_error(actual, predicted))
    print 'R^2:', r2_score(actual, predicted)
    print 'Mean Absolute Error: ', mean_absolute_error(actual, predicted)
    print 'Median Absolute Error: ', median_absolute_error(actual, predicted)
    ab=pd.DataFrame({'Abs_Error': np.absolute(actual-predicted)})
    print '\nQuantiles of Absolute Error:\n', ab.quantile([.75,.85,.9,.95,.97,.98,.99,.995,.998])


"""
Create ROC curve
"""
def create_ROC_curve(actual, preds, file):
    fpr, tpr, thresholds = metrics.roc_curve(actual, preds[:,1])

    fig = plt.figure()
    fig.suptitle("ROC (AUC: %f)" %(metrics.roc_auc_score(actual, preds[:,1])))
    g1 = fig.add_subplot(1,1,1)

    g1.plot(fpr,tpr, color = 'blue', drawstyle='steps-post', label='ROC')
    g1.plot(fpr,fpr, color = 'red', label = 'Random')

    g1.legend(loc='lower right', fancybox = True)

    fig.savefig(file)

def create_precision_recall_curve(actual, preds):
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision, recall, _ = precision_recall_curve(actual, preds)
    average_precision = average_precision_score(actual, preds)

    # Compute micro-average ROC curve and ROC area
    #precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    #average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.show()


"""
HyperTuning SGD Classifier
"""

def HyperTuningSGD(n, alph, SEED,loss, penalty, xs, y, xs_test, y_test, strat=True, print_str=False):
    """
    n: number of cv interations
    alph: list of hypertuning parameters
    SEED: SEED for cv split
    loss: type of loss for SGD Classifier, e.g. 'log'
    penalty: l2 or l1
    xs: dataset to cross-validate on
    y: corresponding target
    xs_test: test set
    y_test: corresponding y

    returns:  dictionary of auc.mean (key): hyperparameter (value)
    """

    out_str = ""
    result = {}
    if print_str:
        print "begin CV loop"

    for hyper in alph:
        auc=[]
        test_auc=[]

        lm1 = SGDClassifier(penalty=penalty,
                             loss=loss,
                             fit_intercept=True,
                             shuffle=True,
                             n_iter=40,
                             n_jobs=-1,
                             verbose=0,
                             random_state=6,
                             alpha=hyper)

        for i in range(n):

            if strat==True:
                sss = cross_validation.StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=(i+1)*SEED)
                for train_index, test_index in sss:
                    x_train, x_cv = xs[train_index], xs[test_index]
                    y_train, y_cv = y[train_index], y[test_index]

            else:
                x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(xs, y, test_size=.2, random_state=(i+1)*SEED)

            lm1.fit(x_train, y_train)
            #preds= lm1.decision_function(x_cv)
            preds= lm1.predict_proba(x_cv)
            auc.append(metrics.roc_auc_score(y_cv, preds[:,1]))
            #test_preds= lm1.decision_function(xs_test)
            test_preds= lm1.predict_proba(xs_test)
            test_auc.append(metrics.roc_auc_score(y_test,test_preds[:,1]))

        auc = np.array(auc)
        print "CV AUC", auc
        print "Test AUC", test_auc
        test_auc= np.array(test_auc)
        out_str += "Parameter: %f AUC: %f (+/- %f) on %d folds\n" % (hyper, auc.mean(), 2*auc.std(), n)
        out_str += "\tTest Performance AUC: %f (+/- %f)\n" % (test_auc.mean(), 2*test_auc.std())
        result[auc.mean()]=hyper
        if print_str:
            print out_str

    return result

"""
HyperTuning SGD Classifier k-fold
"""

def KFoldHyperTuningSGD(k, alph, SEED, loss, penalty, xs, y, xs_test, y_test, print_str=False):
    """
    n: number of cv interations
    alph: list of hypertuning parameters
    SEED: SEED for cv split
    loss: type of loss for SGD Classifier, e.g. 'log'
    penalty: l2 or l1
    xs: dataset to cross-validate on
    y: corresponding target
    xs_test: test set
    y_test: corresponding y

    returns:  dictionary of auc.mean (key): hyperparameter (value)

    """

    out_str = ""
    result = {}
    if print_str:
        print "begin CV loop"

    for hyper in alph:
        auc=[]
        test_auc=[]

        lm1 = SGDClassifier(penalty=penalty,
                             loss=loss,
                             fit_intercept=True,
                             shuffle=True,
                             n_iter=40,
                             n_jobs=-1,
                             verbose=0,
                             random_state=6,
                             alpha=hyper)

        skf = StratifiedKFold(y, n_folds=k, random_state=SEED)
        for train_index, test_index in skf:
            x_train, x_cv = xs[train_index], xs[test_index]
            y_train, y_cv = y[train_index], y[test_index]

            lm1.fit(x_train, y_train)
            preds = lm1.predict_proba(x_cv)
            auc.append(metrics.roc_auc_score(y_cv, preds[:,1]))
            test_preds = lm1.predict_proba(xs_test)
            test_auc.append(metrics.roc_auc_score(y_test, test_preds[:,1]))

        auc = np.array(auc)
        print "CV AUC", auc
        print "Test AUC", test_auc
        test_auc= np.array(test_auc)
        out_str += "Parameter: %f AUC: %f (+/- %f) on %d folds\n" % (hyper, auc.mean(), 2*auc.std(), k)
        out_str += "\tTest Performance AUC: %f (+/- %f)\n" % (test_auc.mean(), 2*test_auc.std())
        result[auc.mean()]=hyper
        if print_str:
            print out_str

    return result
"""
 Lemmer and Stemmer functions
"""

#stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems).encode('utf-8')


# lemmer

#lmtzr = WordNetLemmatizer()
def lem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(lmtzr.lemmatize(item, 'v'))
    return stemmed

def tokenize_lem(text):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = lem_tokens(tokens)
    return ' '.join(stems).encode('utf-8')


# lemmer + Stemmer
def stem_lem_tokens(tokens):
    stemmed = []
    for item in tokens:
        lemmed = lmtzr.lemmatize(item, 'v')
        stemmed.append(stemmer.stem(lemmed))
    return stemmed

def tokenize_lem_stem(text):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = stem_lem_tokens(tokens)
    return ' '.join(stems).encode('utf-8')


# Autocorrector + lemmer + Stemmer
def correct_stem_lem_tokens(tokens):
    stemmed = []
    for item in tokens:
        if len(item)>4:
            item = correct(item)
        lemmed = lmtzr.lemmatize(item, 'v')
        stemmed.append(stemmer.stem(lemmed))
    return stemmed

def tokenize_corr_lem_stem(text):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = correct_stem_lem_tokens(tokens)
    return ' '.join(stems).encode('utf-8')

# Abbrev ampper + lemmer + Stemmer
def map_stem_lem_tokens(tokens, words):
    regex = r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b"
    reobj = re.compile(regex, re.I)
    stemmed = []
    for item in tokens:
        lemmed = lmtzr.lemmatize(item, 'v')
        replaced = reobj.sub(lambda x:words[x.group(0)], lemmed)
        stemmed.append(stemmer.stem(replaced))
    return stemmed

def tokenize_map_lem_stem(text, words):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    #text = str(text.encode('ascii', 'ignore'))
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = map_stem_lem_tokens(tokens, words)
    return ' '.join(stems).encode('utf-8')

def tokenize_map_lem_stem1(text, words):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text = str(text.encode('ascii', 'ignore'))
    text = unicode(text, 'ascii', 'ignore').encode('utf-8').lower()
    tokens = nltk.word_tokenize(text)
    stems = map_stem_lem_tokens(tokens, words)
    return '\n'.join(stems).encode('utf-8')

def tokenize_map_lem_stem3(text, words):
    #text = "".join([ch for ch in text if ch not in string.punctuation])
    text=text.decode('utf-8')
    text = str(text.encode('ascii', 'ignore'))
    text = unicode(text, 'ascii', 'ignore').decode().encode('utf-8').lower()
    text=text.lower()
    tokens = nltk.word_tokenize(text)
    stems = map_stem_lem_tokens(tokens, words)
    return '\n'.join(stems).encode('utf-8')
