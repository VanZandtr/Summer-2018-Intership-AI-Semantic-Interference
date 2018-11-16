from mcrae import *
from oppenheim import *
import glob
import os
import copy
import matplotlib.pyplot as plt
import numpy
import time

# a constant for the directory to store (or load) subject models from
# SUBJ_DIR = 'nat-context-20'
SUBJ_DIR = 'nat-rand40-context-e15-s10'

# a constant for the number of subjects to train
NUM_SUBS = 20

# a constant for the acceptable error rate when training subjects
ACCEPT_ERROR = 0.015

# a constant for the total number of procedures to run, used to calculate the number of repeats
NUM_PROCED = 40
# NUM_REPEATS = NUM_PROCED // NUM_SUBS
# if we are only taking the averages for each cycle, the block order shouldn't matter much
NUM_REPEATS = 1

# activation function
ACT_FUNC = ActivationFunction.SOFTMAX

# probability of adding a random context
RAND_CONTEXT_PROB = .025

def create_subject_group(mcrae_obj, group_size=10, learning_rate=0.1, n_epochs=100, batch_size=50,
                         directory='subjects', activation=ActivationFunction.SOFTMAX):
    """
    Create neural net models representing the concept knowledge of the requested number of random subjects,
    using the distributions described by the McRae norms to generate the individualized training sets. Each
    of the subjects are evaluated are the same test sets. As a side-effect, each subject models and the
    associated training data are saved in Pickled files.
    :type mcrae_obj: McraeNorms
    :param mcrae_obj: An object with Mcrae norms for the concepts used as labels
    :param group_size: The number of subjects to be created
    :param learning_rate: The learning rate to use in training the subject models
    :param n_epochs: The maximum number of epochs to use when training the subject models
    :param batch_size: The number of examples to use in a minibatch when training the subject models
    :param directory: The directory to save the pickled subject models in
    :return: A list of subject models, where each is a SubjectOppenheimSigmoid object (or a subclass of it)
    """

    subject_list = []

    # we'll use the same test set for all subjects
    (test_y, test_x) = make_labeled_data_set(mcrae_obj, 1000, 1.5, RAND_CONTEXT_PROB)
    test_file = os.path.join(directory, 'test_data')
    save_labeled_data(test_x, test_y, test_file)

    n_concepts = len(mcrae_obj.label_list)
    for i in range(group_size):
        print('Creating training data for subject ', i, '...')

        # assume the targets are given as 1D vector of integer labels
        (train_y, train_x) = make_labeled_data_set(mcrae_obj, 50 * n_concepts, random_context=RAND_CONTEXT_PROB)
        train_file = os.path.join(directory, 'train_data_' + format(i,"02d"))
        save_labeled_data(train_x, train_y, train_file)

        print('Training subject ', i, '...')
        subject_file = os.path.join(directory, 'subject_' + format(i,"02d") + '_model.pkl')
        # note, new_subject will also save the model in a file
        new_subject = build_initial_model(train_set=(train_x, train_y), test_set=(test_x, test_y), learning_rate=learning_rate,
                            n_out=n_concepts, n_epochs=n_epochs, batch_size=batch_size, file_name=subject_file,
                            activation=activation, accept_error=ACCEPT_ERROR)
        # inspect the results...
        # test_x = test_x.astype(numpy.float32)             # we must do this to convert test_x from 64 bit float to 32 bit float
        inspect_model(new_subject, mcrae_obj, test_x, test_y)

        subject_list.append(new_subject)
    return subject_list


def blocked_cyclic_naming(subject_model, block_targets, block_feats, num_cycles, verbose=False):
    """ Run the blocked cyclic naming paradigm on a single subject. In this
    paradigm, a small set of pictures is repeatedly presented to the subject.
    A cycle is one full iteration over the pictures. If at any point during the trial
    there is a naming error or a timeout, valid is set to false to indicate that the
    results should be discarded.
    :param subject_model: The pre-trained neural network representing the subject
    :param block_targets: A vector of integer ids indicating the target labels for the examples presented
        to the subject. A block can be homogeneous (pictures from the same category (e.g., clothing,
        animal, furniture, appliance) or heterogeneous (each picture is from a different category).
        Typically, there are four pictures per block
    :param block_feats: A 2D one-hot matrix with row for each example and a column for eaach
    feature. list of the concepts used in this experiment.
    :param num_cycles: how many cycle to run the experiment?
    :return: A list of selection times, one for each presentation of a picture (there
    are len(block) * num_cycles times
    """

    valid = True
    selection_times = []                      # TO-DO? use numpy and size the times?
    # block_feats = build_visual_feature_matrix(block, norm_dict, feat_list)
    for cycle in range(num_cycles):
        # shuffle the targets and features in unison
        permutation = numpy.random.permutation(len(block_targets))
        shuffled_feats = block_feats[permutation]
        shuffled_targets = block_targets[permutation]
        # print('Shuffle: ', shuffled_targets)

        for target_id, concept_vec in zip(shuffled_targets, shuffled_feats):
            # prime the model, return the time and error flags
            boosts, naming_error, timeout = recall_as_learning(subject_model, target_id, concept_vec, verbose=False)
            selection_times.append(boosts)  # calculate time to select the concept
            if naming_error is True or timeout is True:
                valid = False
                if verbose is True:
                    print('\tTime Out: ', timeout, ', Naming Error: ', naming_error)
    return selection_times, valid


def batch_blocked_cyclic_naming(subject_directory, norms_obj, blocks, num_cycles, repeat_num = 5, include_context=False):
    """
    Run the blocked cyclic naming paradigm on a set of subjects, and return the average response times

    :param subject_directory: The directory where pickled subject models are stored
    :param norms_obj: An object with information about the norms. Used to create a feature matrix for the
        examples in the block
    :param blocks: A list of the groups of concepts used in this experiment. Each group (called a
        block) is a list itself. A block can be homogeneous (pictures from the same category
        (e.g., clothing, animal, furniture, appliance) or heterogeneous (each picture is from a
        different category). Typically, there are four pictures per block
    :param num_cycles: how many cycles to run the experiment?
    :param repeat_num: number of times to redo the experiment (with a shuffled block) on same subject
    :param include_context: Determines whether or not context is included
    :return: The average time for each sequential naming event
    """

    # find the names of all subject model files
    subject_file_pattern = os.path.join(subject_directory,'subject_*_model.pkl')
    subject_file_list = glob.glob(subject_file_pattern)

    # assume that all blocks have the same length as the first!!!!
    num_of_events = num_cycles * len(blocks[0])
    # need a row for each subject x each block x repetitions-of-the-block
    time_record = numpy.zeros((len(blocks) * repeat_num * len(subject_file_list), num_of_events))
    print('time record: ', time_record.size, ' ', time_record.shape)

    trial_cnt = 0          # the number of valid trials completed (time outs and naming errors lead to invalid trials
    invalid_cnt = 0

    for seq, subject_model_file in enumerate(subject_file_list):
        with open(subject_model_file, 'rb') as f:          # read the file once
            fresh_subject_model = pickle.load(f)

        for i in range(repeat_num):            # redo the trial with the same set of blocks

            # make sure to start with a fresh model, if we are repeating the experiment on the same subject
            subject_model = copy.deepcopy(fresh_subject_model)

            for block_no, block in enumerate(blocks):                   # for each group of pictures

                # encode the block data in numeric vector/matrix forms
                block_feats = norms_obj.build_visual_feature_matrix(block, include_context=include_context)
                block_targets = encode_target_labels_as_vector(block, norms_obj.label_list)
                # for testing, decode the block...
                # print('Block: ')
                # for x,y in zip(block_feats, block_targets):
                #    decode_feats = decode_feature_matrix([x], norms_obj.feat_list)
                #    print(norms_obj.label_list[y], ': ', decode_feats)

                # familiarize subject with the concepts; this will smooth out some irregularities in times
                # for now, we only present each concept once before starting the experiment. The nature of
                # NNet learning may require us to have more cycles...
                # to improve subject accuracy, we show them the correct answers multiple times
                for _ in range(2):
                    familiar_times, valid = blocked_cyclic_naming(subject_model, block_targets, block_feats, 1)

                # there is a bug that keeps the (more efficient) version below from working
                # additional_training(subject_model,(block_feats,block_targets),n_epochs=5)

                # Shuffle the concepts. We need to make sure that the features and targets are shuffled in unison
                # note, we've changed the code so we are now shuffling each cycle within a trial...
                # permutation = numpy.random.permutation(len(block))
                # shuffled_feats = block_feats[permutation]
                # shuffled_targets = block_targets[permutation]

                # note, the time record index must account for both for the subject no. and the current trial (repeat)
                boosts, valid = blocked_cyclic_naming(subject_model, block_targets, block_feats, num_cycles, verbose=False)
                if valid is True:
                    time_record[trial_cnt] = boosts
                    trial_cnt+=1
                else:
                    invalid_cnt+=1
                # remember, boosts is a list of times, so it is probably not useful to report it...
                print('Subject ', seq, ' , Trial ', i, ', Block #: ', block_no, ', Valid? ', valid)
                # print('Subject ', seq, ' , Trial ', i, ', Avg. Boosts:', numpy.mean(boosts), ', Valid? ', valid)

    print('Summary: ', invalid_cnt, ' discarded trials')

    # take the average along the columns (axis=0?)
    time_record = time_record[:trial_cnt]       # truncate the rows that were unused due to invalid trials
    print('\nTime Record ', time_record.shape, ': ')
    print(time_record)
    mean_times = numpy.mean(time_record, axis=0)

    return mean_times, time_record          # the second value is so that we can save the detailed times

# run some tests
if __name__ == '__main__':
    load_norms = True
    train = True
    inspect = False       # inspects the single model_file model, note train already inpsects each after training is done
    test_taxonomic = False
    test_context = True

    model_file = 'softmax/subject_00_model.pkl'         # used with tests of a single model...
    subject_model_dir = SUBJ_DIR
    # subject_model_dir = 'sigmoid'
    # softmax appears to produce a more a salient semantic blocking effect so far...
    # model_file = 'sigmoid/learned_model.pkl'

    if load_norms is True:
        # read the McraeNorms object from a file
        with open('mcrae_norms.pkl', 'rb') as f:
             mcrae_obj = pickle.load(f)
        print(len(mcrae_obj.feat_list)," features associated with ", len(mcrae_obj.label_list), " concepts")

    # if load_subject is True:
        # with open('learned_model.pkl', 'rb') as f:
        #    subject_model = pickle.load(f)
        # feat_list = read_features()

    if train is True:
        # subject_list = create_subject_group(mcrae_obj, 10, n_epochs=200, directory=subject_model_dir, activation=ActivationFunction.SIGMOID)
        subject_list = create_subject_group(mcrae_obj, NUM_SUBS, directory=subject_model_dir, activation=ACT_FUNC)
        # train_x, train_y, test_x, test_y, n_out = build_train_and_test_sets()
        # train_x, train_y, test_x, test_y, n_out = load_train_and_test_sets()
        #build_initial_model(train_set=(train_x, train_y), test_set=(test_x, test_y), learning_rate=0.1, n_out=n_out,
        #                    file_name=model_file)
        # BUG? It appears that if we use sigmoid with the cost function for softmax, we tend to get very
        # many high activations and very little variation between good and bad classes
        # It is recommended that squared error be used with sigmoid, but then we might push the values towards 0
        # since most of the targets are 0, these will have greater impact on the final cost measure and the
        # target 1 will just be noise...
        # is there work that specifically addresses cost functions for sparse targets?

    if inspect is True:
        # inspect
        subject_model, input_x, input_y = load_model_and_test_data('softmax/test_data', model_file)
        # classifier, input_x, input_y, labelList, featList = loadModelandMakeNewData(normDict)
        inspect_model(subject_model, mcrae_obj, input_x, input_y)

    if test_taxonomic is True:
        hetero_block1 = ['airplane', 'cat', 'chair', 'corn']  # also called mixed condition
        hetero_block2 = ['donkey', 'helicopter', 'shirt', 'drum']
        hetero_block3 = ['fork', 'turtle', 'guitar', 'truck']
        homo_block1 = ['robin', 'ostrich', 'swan', 'penguin']
        homo_block2 = ['donkey', 'horse', 'pig', 'sheep']
        homo_block3 = ['helicopter', 'motorcycle', 'tractor', 'ship']
        homo_block4 = ['shirt', 'skirt', 'sweater', 'dress']
        homo_block5 = ['car', 'truck', 'van', 'bus']
        homo_block6 = ['drum', 'guitar', 'piano', 'saxophone']
        # homo_block7 = ['alligator', 'turtle', 'frog', 'snake']      # no 'snake' in the norms
        # homo_block8 = ['scoop', 'fork', 'spoon', 'ladle']            # no 'scoop' in the norms
        distinct_high_pf = ['unicycle', 'belt', 'bouquet', 'fork', 'porcupine']
        confused = ['perch', 'fridge', 'pheasant', 'curtains', 'gloves', 'cellar', 'chipmunk']

        # heterogenous condition
        # shuffle(hetero_block)
        # print(hetero_block)
        # feat_matrix = mcrae_obj.build_visual_feature_matrix(hetero_block)
        # feat_matrix = build_visual_feature_matrix( confused, norm_dict, feat_map)
        # feat_matrix = feat_matrix.astype(numpy.float32)           # convert from float64 to float32 (for theano)
        # print(decodeFeatureMatrix(feat_matrix, feat_list2))
        # for concept_vec in feat_matrix:
        #    time = selection_time(subject_model, concept_vec)
        #    print(time)
        print('Heterogeneous condition...')
        hetero_blocks = [['donkey', 'shirt', 'car', 'drum'],
                         ['horse', 'skirt', 'truck', 'guitar'],
                         ['pig', 'sweater', 'van', 'piano'],
                         ['sheep', 'dress', 'bus', 'saxophone']] 
        # het_times = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, hetero_blocks, 4, 25)
        het_times, het_details = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, hetero_blocks, 6, NUM_REPEATS, False)

        # the current hetero_block has widely varying boosts, results in subpar figures
        # e..g, airplane ~ 40, cat ~ 0.014, chair ~1.1, corn ~0.04
        # is this due to the use of softmax???
        # or will this average out across many users? Oppenheim runs each simulation 10,000 times and averages...
        # he also drops errors of omission (deadline passes) and commission (wrong word produced)
        print('Times for heterogeneous condition:')
        print(het_times)
        print()

        timestamp = str(int(time.time()))           # get a unique id to use with the output files
        numpy.savetxt("hetero-condition-"+ timestamp +".csv", het_details, delimiter=",", fmt='%1.3f')

        # homogeneous condition
        # shuffle(homo_block)
        # print(homo_block)
        # feat_matrix = mcrae_obj.build_visual_feature_matrix(homo_block)
        print('Homogeneous condition...')
        homog_blocks = [['donkey', 'horse', 'pig', 'sheep'],
                        ['shirt', 'skirt', 'sweater', 'dress'],
                        ['car', 'truck', 'van', 'bus'],
                        ['drum', 'guitar', 'piano', 'saxophone']]
        # homo_times = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, homog_blocks, 4, 25)
        homo_times, homo_details = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, homog_blocks, 6, NUM_REPEATS, False)

        # the current hetero_block has widely varying boosts, results in subpar figures
        # e..g, airplane ~ 40, cat ~ 0.014, chair ~1.1, corn ~0.04
        # is this due to the use of softmax???
        # or will this average out across many users? Oppenheim runs each simulation 10,000 times and averages...
        # he also drops errors of omission (deadline passes) and commission (wrong word produced)
        print('Times for homogeneous condition:')
        print(homo_times)

        numpy.savetxt("homo-condition-"+timestamp+".csv", homo_details, delimiter=",", fmt='%1.3f')

        fig = plt.figure()
        plt.suptitle('Semantic Blocking Experiment')
        # figa = fig.add_subplot(121, title='Heterogenous condition')
        plt.subplot(121)
        plt.title('Heterogenous condition')
        plt.plot(het_times)
        plt.xlabel('Sequence')
        plt.ylabel(('Boosts'))

        # figb = fig.add_subplot(122, title='Homogenous condition')
        plt.subplot(122)
        plt.title('Homogenous condition')
        plt.plot(homo_times)
        plt.xlabel('Sequence')
        plt.ylabel(('Boosts'))
        plt.show()

        # compute the average for each cycle
        het_cycle_means = numpy.mean(het_times.reshape(-1, 4), axis=1)
        hom_cycle_means = numpy.mean(homo_times.reshape(-1, 4), axis=1)
        print(het_cycle_means)
        print(hom_cycle_means)
        fig = plt.figure()
        plt.title('Semantic Blocking Experiment: Cycle Average Comparison')
        plt.plot(numpy.arange(1,5), hom_cycle_means, 'b-', label='Homogeneous condition')
        plt.plot(numpy.arange(1,5), het_cycle_means, 'r-', label='Heterogeneous condition')
        # plt.axis([1,4,38,44])          # typical times between 40 and 43
        plt.legend()
        plt.show()

    if test_context is True:
        # heterogenous condition
        # shuffle(hetero_block)
        # print(hetero_block)
        # feat_matrix = mcrae_obj.build_visual_feature_matrix(hetero_block)
        # feat_matrix = build_visual_feature_matrix( confused, norm_dict, feat_map)
        # feat_matrix = feat_matrix.astype(numpy.float32)           # convert from float64 to float32 (for theano)
        # print(decodeFeatureMatrix(feat_matrix, feat_list2))
        # for concept_vec in feat_matrix:
        #    time = selection_time(subject_model, concept_vec)
        #    print(time)
        print('Unrelated condition...')
        hetero_blocks = [['knife', 'hawk', 'toaster', 'chair'],
                         ['stool_(furniture)', 'wasp', 'plate', 'baton'],
                         ['bucket', 'socks', 'radio', 'violin'],
                         ['worm', 'jacket', 'grapefruit', 'projector']]
        # het_times = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, hetero_blocks, 4, 25)
        het_times, het_details = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, hetero_blocks,
                                                             6, NUM_REPEATS, False)
        print('Times for unrelated condition:')
        print(het_times)
        print()

        timestamp = str(int(time.time()))  # get a unique id to use with the output files
        numpy.savetxt("unrel-condition-" + timestamp + ".csv", het_details, delimiter=",", fmt='%1.3f')

        # shuffle(context_block)
        # print(context_block)
        # feat_matrix = mcrae_obj.build_visual_feature_matrix(context_block)
        print('Labeled context condition...')
        context_blocks = [['knife', 'stool_(furniture)', 'bucket', 'worm'],
                         ['hawk', 'wasp', 'socks', 'jacket'],
                         ['toaster', 'plate', 'radio', 'grapefruit'],
                         ['chair', 'baton', 'violin', 'projector']]
        # context_times = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, homog_blocks, 4, 25)
        lab_context_times, lab_context_details = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj, context_blocks,
                                                                     6, NUM_REPEATS, True)
        print('Times for labeled context condition:')
        print(lab_context_times)
        numpy.savetxt("labeled-context-condition-" + timestamp + ".csv", lab_context_details, delimiter=",", fmt='%1.3f')

        # unlabeled condition uses same blocks as above
        print('Unlabeled context condition...')
        unlab_context_times, unlab_context_details = batch_blocked_cyclic_naming(subject_model_dir, mcrae_obj,
                                                                                 context_blocks, 6, NUM_REPEATS, False)
        print('Times for unlabeled context condition:')
        print(unlab_context_times)
        numpy.savetxt("unlabeled-context-condition-" + timestamp + ".csv", unlab_context_details, delimiter=",", fmt='%1.3f')

        poss_runs = NUM_SUBS * NUM_REPEATS * 4
        print("Successful runs:")
        print('Unrelated: ', het_details.shape[0], ' of ', poss_runs)
        print('Labeled Context: ', lab_context_details.shape[0], ' of ', poss_runs)
        print('Unlabeled Context: ', unlab_context_details.shape[0], ' of ', poss_runs)

        # fig = plt.figure()
        # plt.suptitle('Semantic Blocking Experiment')
        # # figa = fig.add_subplot(121, title='Heterogenous condition')
        # plt.subplot(121)
        # plt.title('Unrelated condition')
        # plt.plot(het_times)
        # plt.xlabel('Sequence')
        # plt.ylabel(('Boosts'))
        #
        # # figb = fig.add_subplot(122, title='Labeled Context condition')
        # plt.subplot(122)
        # plt.title('Labeled context condition')
        # plt.plot(context_times)
        # plt.xlabel('Sequence')
        # plt.ylabel(('Boosts'))
        # plt.show()

        # compute the average for each cycle
        het_cycle_means = numpy.mean(het_times.reshape(-1, 4), axis=1)
        lab_con_cycle_means = numpy.mean(lab_context_times.reshape(-1, 4), axis=1)
        unlab_con_cycle_means = numpy.mean(unlab_context_times.reshape(-1, 4), axis=1)
        print(het_cycle_means)
        print(lab_con_cycle_means)
        print(unlab_con_cycle_means)
        fig = plt.figure()
        plt.title('Semantic Blocking Context Experiment: Cycle Average Comparison')
        plt.plot(numpy.arange(1, 7), lab_con_cycle_means, 'b-', label='Labeled Context condition')
        plt.plot(numpy.arange(1, 7), unlab_con_cycle_means, 'g-', label='Unlabeled Context condition')
        plt.plot(numpy.arange(1, 7), het_cycle_means, 'r-', label='Heterogeneous condition')
        # plt.axis([1,4,38,44])          # typical times between 40 and 43
        plt.legend()
        plt.show()
