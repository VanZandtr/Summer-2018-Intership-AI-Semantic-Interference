'''
created: 6/25/2018
slight change from testingCode.py with taxonomic reordering
lines:
'''
from mcrae import *
from oppenheim import *
import copy
import glob
import numpy
import matplotlib.pyplot as plt
import time

def continuous_naming(subject_directory, norms_obj, blocks, repeat_num=2, verbose=False):

    # find the names of all subject model files
    subject_file_pattern = os.path.join(subject_directory,'subject_*_model.pkl')
    subject_file_list = glob.glob(subject_file_pattern)

    if len(subject_file_list)%len(blocks[0]) is False:
        sys.exit("NUM_SUBS is not a divisible by number of rows (len(blocks[0])")

    # assume that all blocks have the same length as the first!!!!
    #num_of_events = num_cycles * len(blocks[0])
    # need a row for each subject x each block x repetitions-of-the-block
    print('len(blocks): ', len(blocks), 'len(subject_file_list): ', len(subject_file_list))
    
    #time_record = numpy.zeros((len(blocks) * repeat_num * len(subject_file_list), len(blocks[0])))
    time_record = numpy.zeros((repeat_num * len(subject_file_list*4),len(blocks) * len(blocks[0])))

    print('time record: ', time_record.size, ' ', time_record.shape)

    trial_cnt = 0          # the number of valid trials completed (time outs and naming errors lead to invalid trials
    invalid_cnt = 0

            #change text to id numbers
    list = []
    for i in range(0, len(blocks)):
      list.append(encode_target_labels_as_vector(blocks[i], norms_obj.label_list))
    encoded_matrix = numpy.stack(list)

    # encode the block data in numeric vector/matrix forms
    groups = []
    # !!!!! TO-DO: modify this to set up the numbers of groups based on the length of a block.... !!!!
    groups.append(encoded_matrix.transpose())    # first group has one item from each group in each row\
    groups.append(numpy.roll(groups[0],1,0))      # version with rows shifted by 1
    groups.append(numpy.roll(groups[0],2,0))   	  # version with rows shifted by 2
    groups.append(numpy.roll(groups[0],3,0))      # version with rows shifted by 3
  
    timestr = str(int(time.time()))
    text_file = open("Output" + timestr + ".txt", "w")    
    for i in range(0,len(groups)): #loop through groups
        group = groups[i]
        for subj_id in range(0,len(subject_file_list)):
            subject_model_file = subject_file_list[subj_id]
            with open(subject_model_file,'rb') as f:
                fresh_subject_model = pickle.load(f)
                
                
                firstSelectionFlag = 1;
                for i in range(repeat_num): 
                    for row in range(0,len(blocks)):
                        numpy.random.shuffle(group[row])
                    list = group.flatten()# redo the trial with the same set of blocks
                
                    selection_times = []
                    # make sure to start with a fresh model, if we are repeating the experiment on the same subject
                    subject_model = copy.deepcopy(fresh_subject_model)

                    #for block_no, block in enumerate(blocks):
                    feat_matrix = norms_obj.build_visual_feature_matrix_from_ids(list)
                    # use list in place of block_targets
                    # block_targets = encode_target_labels_as_vector(block, norms_obj.label_list)
                    text_file.write('Concept_List:')
                    text_file.write('\t')
                    for target_id, concept_vec in zip(list, feat_matrix):
                        text_file.write(norms_obj.label_list[target_id] + '\t')
                        boosts, naming_error, timeout = recall_as_learning(subject_model, target_id, concept_vec, verbose=False)
                        selection_times.append(boosts)  # calculate time to select the concept
                        valid = True
                        if naming_error is True or timeout is True:
                            valid = False
                            if verbose is True:
                                print('\tTime Out: ', timeout, ', Naming Error: ', naming_error)

                    if valid is True:
                        time_record[trial_cnt] = selection_times
                        
                        if firstSelectionFlag is 1:
                            print('time_record of first pick: ', time_record[trial_cnt][0], ', Word: ', norms_obj.label_list[target_id])
                            #print('time_record: ', time_record[trial_cnt])
                            firstSelectionFlag = 0
                        
                        trial_cnt+=1
                        print('Subject ', subj_id, ' , Trial ', i, ', Valid? ', valid, ', Word: ', norms_obj.label_list[target_id])
                        
                    else:
                        invalid_cnt+=1
                        # remember, boosts is a list of times, so it is probably not useful to report it...
                        print('Subject ', subj_id, ' , Trial ', i, ', Valid? ', valid, ', Word: ', norms_obj.label_list[target_id])
                        # print('Subject ', seq, ' , Trial ', i, ', Avg. Boosts:', numpy.mean(boosts), ', Valid? ', valid)
                
                    text_file.write('\n')
                text_file.write('\n')
                print('\n')
                
    text_file.close()
    print('Summary: ', invalid_cnt, ' discarded trials')

    # take the average along the columns (axis=0?)
    time_record = time_record[:trial_cnt]       # truncate the rows that were unused due to invalid trials
    print('\nTime Record ', time_record.shape, ': ')
    print(time_record)
    mean_times = numpy.mean(time_record, axis=0)
    quartile_times = numpy.mean(mean_times.reshape(-1, len(blocks[0])), axis=1)
    return mean_times, time_record, quartile_times 


# run some tests
if __name__ == '__main__':

    subject_directory = 'nat-rand40-context-e15-s10'
    with open('mcrae_norms.pkl', 'rb') as f:
        mcrae_obj = pickle.load(f)
    print(len(mcrae_obj.feat_list)," features associated with ", len(mcrae_obj.label_list), " concepts")

    
    #TEST 1
    print("Running continuous paradigm experiment...")

    blocks = [['donkey', 'horse', 'pig', 'sheep'],
                        ['shirt', 'skirt', 'sweater', 'dress'],
                        ['car', 'truck', 'van', 'bus'],
                        ['drum', 'guitar', 'piano', 'saxophone']]
    mean_times, time_record, quartile_times = continuous_naming(subject_directory, mcrae_obj, blocks)
    
    timestamp = str(int(time.time()))
    numpy.savetxt("continuous_mean_times-"+ timestamp +".csv", mean_times, delimiter=",", fmt='%1.3f')
    numpy.savetxt("continuous_time_record-"+ timestamp +".csv", time_record, delimiter=",", fmt='%1.3f')
    numpy.savetxt("continuous_quartile_times-"+ timestamp +".csv", quartile_times, delimiter=",", fmt='%1.3f')

    plt.plot(mean_times)
    plt.show()
    
    plt.plot(quartile_times)
    plt.show()
    
    '''
    #TEST 2
    print("Running continuous paradigm experiment...")

    blocks = [['shirt', 'shirt', 'shirt', 'shirt'],
        ['skirt', 'skirt', 'skirt', 'skirt'],
        ['sweater', 'sweater', 'sweater', 'sweater'],
        ['dress', 'dress', 'dress', 'dress']]
    mean_times, time_record, quartile_times = continuous_naming(subject_directory, mcrae_obj, blocks)
    
    timestamp = str(int(time.time()))
    numpy.savetxt("continuous_mean_times-"+ timestamp +".csv", mean_times, delimiter=",", fmt='%1.3f')
    numpy.savetxt("continuous_time_record-"+ timestamp +".csv", time_record, delimiter=",", fmt='%1.3f')
    numpy.savetxt("continuous_quartile_times-"+ timestamp +".csv", quartile_times, delimiter=",", fmt='%1.3f')

    plt.plot(mean_times)
    plt.show()
    
    plt.plot(quartile_times)
    plt.show()
    '''