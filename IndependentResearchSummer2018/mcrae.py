import csv
import numpy
from random import *
import timeit
import pickle
import os
import sys

CONTEXT_CHOICES = ['context_concert', 'context_fishing_trip', 'context_breakfast', 'context_hike']

class McraeNorms(object):
    """
    A class for storing the essential Mcrae norm information
    """

    def __init__(self, norms_file='..\Macrae-Norms-CONCS_FEATS_concstats_brm.csv',
                 feats_file='..\Macrae-Norms-FEATS_brm.csv'):
        """
        Initialize the object from the features file and the concepts-features file
        :return:
        """
        self.norms = self.read_norms(norms_file)
        self.label_list = list(set(self.norms.keys()))    # the unique set of concepts that we have norms for
        self.feat_list = self.read_features(feats_file)   # the features we have ascribed to our norms

    @staticmethod
    def read_norms(norms_file):
        """
        Read the norms from a file and build a dictionary for them
        the dictionary keys are concept, and each key is associated with a
        list of norms, where each norm is a dictionary with values for
        feature, wblabel, brlabel, freq, a calculated probability,
        a boolean to indicate if the feature is distinguishing, and
        a measure of distinctiveness
        :return: A dictionary of norms, indexed by concept name
        """
        normDict = {}
        with open(norms_file, 'r') as csvfile:
            normReader = csv.reader(csvfile)
            next(normReader,None)            # skip the header
            for row in normReader :
                anorm = {}
                anorm['feature'] = row[1]
                anorm['wblabel'] = row[2]
                anorm['brlabel'] = row[5]
                anorm['freq'] = int(row[6])
                anorm['prob'] = int(row[6]) / 30.0
                anorm['disting'] = False
                if row[10] == 'D':
                    anorm['disting'] = True
                anorm['distinct'] = float(row[11])
                concept = row[0]
                if concept in normDict:
                    featList = normDict[concept]   # returns a reference to the featList, not a copy
                    featList.append(anorm)         # adds to the featList in normDict
                else:
                    featList = [anorm]
                    normDict[concept] = featList
                    # featList.append(anorm)
        # for (concept,featList) in normDict.items():
        #     print(concept)
        #     for norm in featList:
        #         print('\t', norm)
        return normDict

    @staticmethod
    def read_features(feats_file):
        """
        Read the list of features and return them as a list
        :param feats_file:
        :return:
        """
        feat_list = []
        with open(feats_file, 'r') as csvfile:
            featReader = csv.reader(csvfile)
            next(featReader,None)             # skip the header
            for row in featReader:
                feat_list.append(row[0])
        return feat_list


    def sample_concept(self, concept, unique_thresh = 0.2):
        """
        Sample the features of a specific concept from the norm dictionary
        randomly decides for each feature if it is present and adds it to a list
        The unique_thresh is the minimum sum of disting values for selected
        features for the concept

        :param concept: The concept to sample features for
        :param unique_thresh: The minimum acceptable value for the sum of 'distinct' values of the
            sampled concept
        :return:
        """
        unique_level = 0
        feat_list = self.norms[concept]
        result = []
        is_discrim = False
        # continue adding features until either we've exceeded the threshold or have most (80%) of the features
        while unique_level < unique_thresh and len(result) < (0.8 * len(feat_list)):
            for norm in feat_list:
                if random() < norm['prob'] and norm['feature'] not in result :
                    result.append(norm['feature'])
                    unique_level = unique_level + norm['distinct']
        return result


    def build_visual_feature_matrix(self, concept_list, include_context=False):
        """ For each concept in the list, encode it as a feature vector using
        all of the visual features, and no other features. This is intended
        to simulate the features that might be activated when presented with
        a picture of the object in question

        :param concept_list: A list of concepts
        :param include_context: Determines whether or not the subject should learn a context
        :return: A 2D array of 0/1s with a separate row for each concept and column for each feature
        """

        # we use the BR_label to determine whether a feature is visual
        # for now, we are assuming that visual-motion features will not be activated by pictures...a
        if include_context:
            feat_types=['taxonomic', 'visual-colour', 'visual-form_and_surface', 'context']
        else:
            feat_types = ['taxonomic', 'visual-colour', 'visual-form_and_surface']

        disc_data = []                     # the examples as a 2D array of discrete features
        for concept in concept_list:
            one_ex = []
            poss_feats = self.norms[concept]
            for norm in poss_feats:
                if norm['brlabel'] in feat_types:
                    one_ex.append(norm['feature'])
            disc_data.append(one_ex)
        # print(disc_data)

        value_map = make_discrete_to_int_mapping(self.feat_list)
        feat_vector = encode_feature_matrix(disc_data, value_map)
        return feat_vector

    def build_visual_feature_matrix_from_ids(self, concept_list, include_context=False):
        """ For each concept in the list, encode it as a feature vector using
        all of the visual features, and no other features. This is intended
        to simulate the features that might be activated when presented with
        a picture of the object in question

        :param concept_list: A list of concepts
        :param include_context: Determines whether or not the subject should learn a context
        :return: A 2D array of 0/1s with a separate row for each concept and column for each feature
        """

        # we use the BR_label to determine whether a feature is visual
        # for now, we are assuming that visual-motion features will not be activated by pictures...a
        if include_context:
            feat_types=['taxonomic', 'visual-colour', 'visual-form_and_surface', 'context']
        else:
            feat_types = ['taxonomic', 'visual-colour', 'visual-form_and_surface']

        disc_data = []                     # the examples as a 2D array of discrete features
        for concept_id in concept_list:
            one_ex = []
            poss_feats = self.norms[self.label_list[concept_id]]
            for norm in poss_feats:
                if norm['brlabel'] in feat_types:
                    one_ex.append(norm['feature'])
            disc_data.append(one_ex)
        # print(disc_data)

        value_map = make_discrete_to_int_mapping(self.feat_list)
        feat_vector = encode_feature_matrix(disc_data, value_map)
        return feat_vector

    def get_full_features(self, concept_list):
        """For each concept in the list, return the full set of features that describe the
        concept, regardless of production frequency
        :param concept_list:
        :return: A 2D list of feature names, with one list per concept
        """
        disc_data = []                     # the examples as a 2D array of discrete features
        for concept in concept_list:
            one_ex = []
            full_feats = self.norms[concept]
            for norm in full_feats:
                one_ex.append(norm['feature'])
            disc_data.append(one_ex)
        return disc_data


    def get_taxonomic_and_key_distinct_features(self, concept_list):
        """ For each concept in the list, return a pair of features such that the first feature
        is taxonomic (and shared with many concepts) and the second is as distinctive as possible
        without having a low production frequency
        :param concept_list:
        :return:
        """
        disc_data = []                     # the examples as a 2D array of discrete features
        for concept in concept_list:
            one_ex = []
            feat_sum = []
            full_feats = self.norms[concept]
            disting_prob = 0
            most_disting = 'n/a'
            for norm in full_feats:
                if norm['brlabel'] == 'taxonomic':
                    one_ex.append(norm['feature'])
                if norm['disting'] is True:
                    if norm['prob'] > disting_prob:
                        most_disting = norm['feature']
                        disting_prob = norm['prob']
            one_ex.append(most_disting)
            disc_data.append(one_ex)
        return disc_data


# this method is not currently called, but is kept in case it might be useful later
    def save_lists_to_file(self):
        """
        Output the labels and features lists as human-readable CSV files
        :return:
        """
        with open("labels.csv", 'w', newline='') as csvfile:
            labelWriter = csv.writer(csvfile, delimiter=",")
            for str in self.label_list:
                labelWriter.writerow([str])
        with open("features.csv", 'w', newline='') as csvfile:
            featWriter = csv.writer(csvfile, delimiter=",")
            for str in self.feat_list:
                featWriter.writerow([str])


# given a list of distinct labels, returns a dictionary that
# maps labels to unique integers. This can be used to build a
# one-hot-vector
def make_discrete_to_int_mapping(labels):
    labelMap = {}
    assert isinstance(labels, list)
    for i,item in enumerate(labels):
        labelMap[item] = i
    return labelMap


# this function is deprecated...
# given a list of discrete values and a map of these values
# to unique integer ids, creates a 2D matrix with one-hot encoding
# for each label
# def encode_discrete_vector(values, valMap):
#     encodedMat = numpy.zeros((len(values),len(valMap)))
#     for rowNum,val in enumerate(values):
#         valId = valMap[val]
#         encodedMat[rowNum, valId] = 1
#     return encodedMat


def encode_target_labels_as_vector(target_labels, label_list):
    """
    Given a list of target labels and a cannoncial list of the full set of available labels,
    creates an integer vector that encodes the targets based on their position in the label_list
    :param target_labels: The discrete labels that we want to convert to integer
    :param label_list:
    :return:
    """
    label_map = make_discrete_to_int_mapping(label_list)
    target_vector = numpy.zeros(len(target_labels), dtype=numpy.int32)
    for idx, concept in enumerate(target_labels):
        target_vector[idx] = label_map[concept]
    return target_vector


def encode_feature_matrix(examples, label_map):
    """
    Given a 2D array of binary feature labels, creates a list
    feature vectors where the corresponding position for each
    selected feature is 1, and all other features are 0

    :param examples: A 2D array with each row specifying the symbolic features for an example
    :param label_map: A dictionary that maps feature labels to unique integers
    :return:
    """
    # note, numpy.zeros defaults to float64, but theano uses float32. Thus, we'll create in float32 and avoid future conversions...
    encodedEx = numpy.zeros((len(examples),len(label_map)), dtype=numpy.float32)
    for rowNum,ex in enumerate(examples):
        for feat in ex:
            featId = label_map[feat]
            encodedEx[rowNum, featId] = 1
    return encodedEx


# given a matrix of binary feature vectors for a set of examples
# and a list of ordered feature labels used to produce it, decodes the
# data into a array of examples with an array of feature names
def decode_feature_matrix(encodedEx, labels):
    examples = []
    for ex in encodedEx:
        newRow = []
        for col, feat in enumerate(ex):
            if feat == 1:
                newRow.append(labels[col])
        examples.append(newRow)
    return examples


def make_labeled_data_set(norms_obj, size, unique_thresh=0.2, random_context=0.0):
    """
    Build a set of labeled examples and return as a pair
    of a 1D vector containing integer target labels and a
    2D vector with binary (0/1) classification of features.
    The unique_thresh is used by sample concept to decide
    if the sampled features are discriminating enough

    :type norms_obj: McraeNorms
    :param norms_obj:
    :param size:
    :param unique_thresh:
    :type random_context: float
    :param random_context: The chance 0<x<1 that an example should be assigned a random context
    :return: Tuple(,)
    """
    discData = []
    int_labels = []
    labelMap = make_discrete_to_int_mapping(norms_obj.label_list)
    valueMap = make_discrete_to_int_mapping(norms_obj.feat_list)

    rand_context_count = 0

    for i in range(size):                       # each iteration creates one training example
        concept = choice(norms_obj.label_list)
        # print(concept, ": ", labelMap[concept])
        oneEx = norms_obj.sample_concept(concept, unique_thresh=unique_thresh)
        # print(oneEx)

        # if there are no contexts already, add one with a certain probability
        if random_context > 0 and random() < random_context:
            # print('Add something to ', concept, '...')
            no_context = True
            for c in CONTEXT_CHOICES:
                if c in oneEx:
                    no_context = False
                    break
            if no_context:
                chosen = choice(CONTEXT_CHOICES)
                oneEx.append(chosen)
                rand_context_count += 1
                # print('Added context ', chosen, ' to ', concept)

        int_labels.append(labelMap[concept])
        discData.append(oneEx)

# vecLabels = encodeDiscreteVector(discLabels,labelMap)
    vec_data = encode_feature_matrix(discData, valueMap)
    print('   Added ', rand_context_count, ' random contexts')
    return int_labels, vec_data


def build_train_and_test_sets(norms_obj, random_context=0.0):
    """
    Reads a set of feature norms and then creates a training set and test set using them. Currently uses
    50 examples per category/item in the training set and 1000 total examples in the test set, but it may make
    sense to parameterize them
    :type norms_obj: McraeNorms
    :param norms_obj:
    :return:
    """

    print("Building the training and test sets...")

    # assume the targets are given as 1D vector of integer labels
    (train_y, train_x) = make_labeled_data_set(norms_obj, 50 * len(norms_obj.label_list), random_context)
    (test_y, test_x) = make_labeled_data_set(norms_obj, 1000, 1.5, random_context)

    # it should be more efficient to save the data sets as Pickled data instead of as CSV
    # we can always output to text later if we have the need for humans to look at the data...
    # numpy.savetxt("train_y.csv", train_y, delimiter=",", fmt='%1d')
    # numpy.savetxt("train_x.csv", train_x, delimiter=",", fmt='%1d')
    # numpy.savetxt("test_y.csv", test_y, delimiter=",", fmt='%1d')
    # numpy.savetxt("test_x.csv", test_x, delimiter=",", fmt='%1d')
    with open("train_y.pkl", 'wb') as f:
        pickle.dump(train_y, f)
    with open("train_x.pkl", 'wb') as f:
        pickle.dump(train_x, f)
    with open("test_y.pkl", 'wb') as f:
        pickle.dump(test_y, f)
    with open("test_x.pkl", 'wb') as f:
        pickle.dump(test_x, f)

    return train_x, train_y, test_x, test_y


def save_labeled_data(data_x, data_y, file_name):
    """
    Saves labeled data to a file. This abstraction allows me to experiment with the most efficient
    way to save data. Currently, we convert the data to small int formats before pickling in order
    to be safe.
    :param data_x: The feature data in a 2D matrix
    :param data_y: The label ids in an integer vector
    :param file_name: The file name (minus extension) where the data will be saved
    :return:
    """
    comp_data_x = data_x.astype(numpy.int8)   # this should be safe since x is just 0's and 1's
    # data_y is just a list, so we can't chnage the type...
    # comp_data_y = data_y.astype(numpy.uint16)  # this will be safe as long as there are fewer than 2^16 target labels
    # data_x_and_y = (comp_data_x, comp_data_y)
    data_x_and_y = (comp_data_x, data_y)
    out_file = file_name + '.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(data_x_and_y, f, pickle.HIGHEST_PROTOCOL)


def load_labeled_data(file_name):
    """
    Reads labeled data from a file. This abstraction allows me to experiment with the most efficient
    way to save data. Since the data is currently saved in the most efficient numeric formats, we need to convert
    it back into a form that is usable by Theano before returning it
    :param file_name: The file name (minus extension) where the data will be read from
    :return:
    """
    in_file = file_name + '.pkl'
    with open(in_file, 'rb') as f:
        data_x_and_y = pickle.load(f)
    #(comp_data_x, comp_data_y) = data_x_and_y
    (comp_data_x, data_y) = data_x_and_y
    data_x = comp_data_x.astype(numpy.float32)     # widen to 32 bit floats (for theano)
    # data_y is just a list, so we can't change the type...
    # data_y =  comp_data_y.astype(numpy.float32)    # not sure if I need this for theano, or if I can cast to an int...
    return data_x, data_y


# TO_DO: Replace this with reading from the pickled files
def load_train_and_test_sets():
    print("Loading training and test data sets from files...")
    # theano requires 32 bit floats....
    train_x = numpy.loadtxt('train_x.csv', dtype=numpy.float32, delimiter=',')
    train_y = numpy.loadtxt('train_y.csv', dtype=numpy.int, delimiter=',')
    test_x = numpy.loadtxt('test_x.csv', dtype=numpy.float32, delimiter=',')
    test_y = numpy.loadtxt('test_y.csv', dtype=numpy.int, delimiter=',')

    # calculate the number of classes/labels for use in specifying the NNet params
    with open('labels.csv','r') as csvfile:
        labelReader = csv.reader(csvfile)
        count = 0
        for row in labelReader:
            count += 1

    # print(input_y)
    # print(input_x)
    return train_x, train_y, test_x, test_y, count


# Deprecated: This should no longer be needed now that we have packaged labels and features with the Norms object...
def load_labels_and_features():
    labelList = []
    with open('labels.csv','r') as csvfile:
        labelReader = csv.reader(csvfile)
        for row in labelReader:
            labelList.append(row[0])

    featList = []
    with open('features.csv','r') as csvfile:
        featReader = csv.reader(csvfile)
        for row in featReader:
            featList.append(row[0])
    return labelList, featList


def load_model_and_test_data(test_data_file, model_file='learned_model.pkl'):

    with open(model_file, 'rb') as f:
        classifier = pickle.load(f)

    # theano requires 32 bit floats....
    # input_x = numpy.loadtxt('test_x.csv', dtype=numpy.float32, delimiter=',')
    # input_y = numpy.loadtxt('test_y.csv', dtype=numpy.int, delimiter=',')
    # print(input_y)
    # print(input_x)

    input_x, input_y = load_labeled_data(test_data_file)
    return classifier, input_x, input_y


def load_model_and_make_new_data(normDict):
    with open('learned_model.pkl', 'rb') as f:
        classifier = pickle.load(f)

    labelList, featList = load_labels_and_features()

    (input_y, input_x) = make_labeled_data_set(normDict, labelList, featList, 1000, 1.5)
    # Theano expects the inputs to be 32 bit...
    input_x = input_x.astype(numpy.float32)
    return classifier, input_x, input_y, labelList, featList

def jaccard_distance(c1, c2):
    """ Calculates the Jaccard distance between the features for two concepts.
    This is the intersection of the the two sets divided by the union.
    :param c1: first concept
    :param c2: second concept
    :return: A float indicating the distance
    """

    # TO-DO!!!!

# top-level code
if __name__ == '__main__':

    create = True
    compress = False
    feats = True

    if create is True:
        # create a brand new McRae Norms object by reading from the CSV files...
        if os.path.exists('mcrae_norms.pkl'):
            overwriteAns = input('Do you wish to overwrite the existing file? (Y/N) ')
            if overwriteAns is 'N':
                print('Canceling program...')
                sys.exit()
        mcrae_obj = McraeNorms()
        print(len(mcrae_obj.feat_list)," features associated with ", len(mcrae_obj.label_list), " concepts")
        print(mcrae_obj.label_list)
        print(mcrae_obj.feat_list[0:50])
        with open('mcrae_norms.pkl', 'wb') as f:
            pickle.dump(mcrae_obj, f)

    if compress is True:
        ignore, test_x, test_y = load_model_and_test_data('softmax/test_data.pkl')
        print(test_x.shape, ', ', len(test_y))
        save_labeled_data(test_x, test_y, 'softmax/comp_test_data')
        test_x, text_y = load_labeled_data('softmax/comp_test_data')
        print(test_x.shape, ', ', len(test_y))

    print(mcrae_obj.get_full_features(['baton', 'bucket', 'chair', 'grapefruit', 'hawk', 'jacket', 'knife', 'plate', 'projector', 'radio', 'socks', 'stool_(furniture)', 'toaster', 'violin', 'wasp', 'worm']))
    if  feats is True:
        print(mcrae_obj.get_taxonomic_and_key_distinct_features(['baton', 'bucket', 'chair', 'grapefruit', 'hawk', 'jacket', 'knife', 'plate', 'projector', 'radio', 'socks', 'stool_(furniture)', 'toaster', 'violin', 'wasp', 'worm']))

    # test opening the norms objec
    # with open('mcrae_norms.pkl', 'rb') as f:
    #     mcrae_temp = pickle.load(f)
    # print(len(mcrae_temp.feat_list)," features associated with ", len(mcrae_temp.label_list), " concepts")
    # print(mcrae_temp.label_list)
    # print(mcrae_obj.feat_list[0:50])

    # read a previously created model and generate new test data for it...
    # classifier, input_x, input_y, label_list, featList = load_model_and_make_new_data(norm_dict)
    # inspectModel(classifier, input_x, input_y, label_list, featList)

    # read a previously created model and examine its performance on its test data
    # classifier, input_x, input_y, labelList, featList = loadModelAndTestData()
    # inspectModel(classifier, input_x, input_y, labelList, featList)


    # print(train_y)
    # print(train_x)

    # for i in range(0,5):
    #    print()
    #    concept = 'elk'
    #    print('Typical ', concept, ':')
    #    result = sampleConcept(normDict, concept)
    #    print(result)
    # print(featList)
    # print(makeDiscreteToIntMapping(featList))


# test functions...
def testEncoding():
    testFeat = ['A', 'B', 'C', 'D']
    testEx = [['A','B'], ['B','D'], ['A','C','D'], ['A','B','C','D'],[]]
    labelMap = make_discrete_to_int_mapping(testFeat)
    encoded = encode_feature_matrix(testEx, labelMap)
    print(encoded)
    decoded = decode_feature_matrix(encoded, testFeat)
    print(decoded)
