import theano
import theano.tensor as T
import numpy
import math
import heapq
from mcrae import *
from enum import Enum

# boosting_rate: Oppenheim = 1.01, Seip = 1.06
# higher rates can reduce the large values that occur for some concepts, but may "erase" the effects of others
boosting_rate = 1.06
# threshhold: Oppenheim = 1, Seip = 1
boosting_thresh = 10
# deadline (time-out): Oppenheim = 100
deadline = 100
# learning rate: Oppenheim = 0.75 (but this is way too big for my models)


class ActivationFunction(Enum):
    SIGMOID = 1
    SOFTMAX = 2


class SubjectOppenheimSigmoid(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # BUG? should we initialize W and b with random values?
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        # act_func is a parameter, so we can use the same code with different activation functions
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # note, Oppenheim (2010) uses the logistic function...

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def loss_function(self, y):
        """Return the mean square error (MSE) of the prediction
        of this model under a given target distribution. MSE is the typical
        error used with regression and the sigmoid. For each example subtract the
        target from the prediction and square the result

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # y is give as a vector of target integer labels, but in order to find square error, we need to convert
        # these into one-hot vectors!
        # option 1: theano.tensor.extra_ops.to_one_hot(seq, num_classes))
        # option 2: tt.eq(seq.reshape((-1, 1)), tt.arange(num_classes))

        # return T.mean(T.pow(self.p_y_given_x - theano.tensor.extra_ops.to_one_hot(y, self.p_y_given_x.shape[1]), 2))

        # let's try sum
        return T.sum(T.pow(self.p_y_given_x - theano.tensor.extra_ops.to_one_hot(y, self.p_y_given_x.shape[1]), 2))

        # this will run, but is not what we want for sigmoid
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class SubjectOppenheimSoftMax(SubjectOppenheimSigmoid):

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # set W, b, y_pred, input, and params the same as super class
        super().__init__(input, n_in, n_out)

        # override the p_y_given_x to use softmax
        # note, Oppenheim (2010) uses the logistic function...
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)


    def loss_function(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution. This is a good
        loss function for softmax and multiclassification problems

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

# Each example has 2526 features, each of which is 4 bytes. Thus, a single
# example requires about 10K of memory. A batch size of 50 results in processing
# about 0.5 MB at a time.
# In test runs, batch=50, learn=0.1 reached 0.4% error in 110 epochs
# batch = 100, learn=0.13 took 195 epochs
def build_initial_model(train_set, test_set, n_out, learning_rate=0.1, n_epochs=150, batch_size=50,
                        file_name='learned_model.pkl', activation=ActivationFunction.SOFTMAX, accept_error=0.005):
    """
    Use stochastic gradient descent optimization of a log-linear
    model for Mcrae norms

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :rtype: SubjectOppenheimSigmoid
    """

    (train_x, train_y) = train_set
    (test_x, test_y) = test_set

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.shape[0] // batch_size

    # make datasets compatible with Theano
    train_set_x = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX),
                                borrow=True)
    train_set_y_temp = theano.shared(numpy.asarray(train_y, dtype=theano.config.floatX),
                                     borrow=True)
    train_set_y = T.cast(train_set_y_temp, 'int32')
    test_set_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX),
                               borrow=True)
    test_set_y_temp = theano.shared(numpy.asarray(test_y, dtype=theano.config.floatX),
                                    borrow=True)
    test_set_y = T.cast(test_set_y_temp, 'int32')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    if activation == ActivationFunction.SIGMOID:
        classifier = SubjectOppenheimSoftMax(input=x, n_in=train_x.shape[1], n_out=n_out)
    elif activation == ActivationFunction.SOFTMAX:
        classifier = SubjectOppenheimSoftMax(input=x, n_in=train_x.shape[1], n_out=n_out)
    else:
        print('Undefined activation function: Using softmax')
        classifier = SubjectOppenheimSoftMax(input=x, n_in=train_x.shape[1], n_out=n_out)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.loss_function(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # validate_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # if epoch % 5 == 0 and minibatch_index == 0:
            #     print('\t\tCost at ', epoch, ':', minibatch_index, ' - ', minibatch_avg_cost)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # if (iter + 1) % validation_frequency == 0:
            #     # compute zero-one loss on validation set
            #     validation_losses = [validate_model(i)
            #                          for i in range(n_valid_batches)]
            #     this_validation_loss = numpy.mean(validation_losses)
            #
            #     print(
            #         'epoch %i, minibatch %i/%i, validation error %f %%' %
            #         (
            #             epoch,
            #             minibatch_index + 1,
            #             n_train_batches,
            #             this_validation_loss * 100.
            #         )
            #     )
            #
            #     # if we got the best validation score until now
            #     if this_validation_loss < best_validation_loss:
            #         #improve patience if loss improvement is good enough
            #         if this_validation_loss < best_validation_loss * \
            #                 improvement_threshold:
            #             patience = max(patience, iter * patience_increase)
            #
            #         best_validation_loss = this_validation_loss
            #         # test it on the test set
            #
            #         test_losses = [test_model(i)
            #                        for i in range(n_test_batches)]
            #         test_score = numpy.mean(test_losses)
            #
            #         print(
            #             (
            #                 '     epoch %i, minibatch %i/%i, test error of'
            #                 ' best model %f %%'
            #             ) %
            #             (
            #                 epoch,
            #                 minibatch_index + 1,
            #                 n_train_batches,
            #                 test_score * 100.
            #             )
            #         )
            #
            #         # save the best model
            #         with open('best_model.pkl', 'w') as f:
            #             pickle.dump(classifier, f)

            # if patience <= iter:
            #     done_looping = True
            #     break

        # periodically test the results (every 5 epochs)
        if (epoch % 5) == 0:
            test_losses = [test_model(i)
                           for i in range(n_test_batches)]
            test_score = numpy.mean(test_losses)

            print(
                (
                    '     epoch %i, test error of  %4.1f%%'
                ) %
                (
                    epoch, test_score * 100.
                )
            )

            if test_score < accept_error:  # acceptable error rate
                done_looping = True

        # periodically save the model
        if (epoch % 50) == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(classifier, f)

    end_time = timeit.default_timer()

    print('The code ran for %d epochs and %d seconds, with %f epochs/sec' % (
        epoch, end_time - start_time, 1. * epoch / (end_time - start_time)))

    # save the final model
    with open(file_name, 'wb') as f:
        pickle.dump(classifier, f)

    return classifier


def inspect_model(classifier, norms_obj, input_x, input_y):
    """
    Run a data set through the classifier and report on each classification error. The norms_obj
    provides the normalize lists of features and labels that are used to decode the input matrices.
    :type classifier: SubjectOppenheimSigmoid
    :param classifier:
    :type norms_obj: McraeNorms
    :param norms_obj:
    :param input_x:
    :param input_y:
    :return:
    """

    # compile a predictor function
    # this function requires an array-like object as input
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # if the data is a theano variable, this is how to get the array/matrix...
    # input_x = test_set_x.get_value()

    predicted_values = predict_model(input_x)
    print("Full set of prediction errors found in in the input data set:")
    errCnt = 0
    for index in range(len(input_x)):
        # only print the mistakes...
        if input_y[index] != predicted_values[index]:
            # since decodeFeatureMatrix expects a 2D array of examples, we must
            # wrap a single example in an list
            print(index, ":", decode_feature_matrix([input_x[index]], norms_obj.feat_list))
            print("\tTarget:    ", norms_obj.label_list[input_y[index]])
            print("\tPrediction:", norms_obj.label_list[predicted_values[index]])
            errCnt += 1
    print()
    print(errCnt, "/", len(input_x), " prediction errors")


def selection_time(subject_model, concept_vec, verbose=False):
    """ Compute the selection time for recognizing the given concept in a singel trial.
        We must do this on a one-off basis, because we need to modify the model after each
        recognition, in order to simulate priming. We use Oppenheim's formula:
        t_select = log beta ( thresh / (act_max - mean(act_others))
        using algebra, the denominator becomes
        (n_output * (act_max - avg(act_all))/(n_output - 1)
        beta = 1.06 (boosting rate value from Seip), thresh = 1 (value from Oppenheim)

    :param subject_model: The pre-trained neural network representing the subject
    :param concept_vec: A 0/1 vector of features describing the concept presented
    :return: A float representing the time to produce the concept name
    """

    # compile a function to retrieve the activation values
    get_model_activations = theano.function(
        inputs=[subject_model.input],
        outputs=subject_model.p_y_given_x)

    # the choice of k should take the activation function into account.
    # With softmax, when you have 541 features, the average is usually very small, e.g., 0.00184843
    # this means the time is almost completely dependent on the max value, and the competitors have no effect
    # For sigmoid, many of the values are very large (top 100 average at 0.89-0.92, top 400 average 0.81-0.84)
    # two possible solutions (for softmax):
    # 1) use the top k competitor activations to get the average (k could be = 1)
    # 2) only use competitors that were part of the naming experiment (seems unrealistic)
    # avg_val = numpy.mean(activations)             # values are too small...

    k = 5              # good value for softmax
    # k = 400  # possible value for sigmoid?

    activations = get_model_activations([concept_vec])
    activations = activations[0]  # get the first vector of results, since we only passed in one example
    activations = heapq.nlargest(k, activations)  # get the top-k activations values
    # print("Act = ", activations)         # print out the largest activations
    max_val = numpy.max(activations)
    avg_val = numpy.mean(activations)
    n_output = len(activations)
    diff = n_output * (max_val - avg_val) / (n_output - 1)  # FIX? it would be cleaner just to remove the first item before averaging

    if verbose is True:
        print("Max: ", max_val, ", Avg: ", avg_val, ", Num: ", n_output, ", Diff: ", diff)
    time = math.log(boosting_thresh / diff, boosting_rate)

    # Note, Oppenheim (2010) has a parameter omega (deadline) that is the maximum number of boosts
    # if this parameter is exceeded, we have an error of omission
    # we account for this in the recall_as_learning function
    return time


def additional_training(subject_model, train_set, learning_rate=0.1, n_epochs=5, batch_size=50, accept_error=0.01, verbose = False):
    """
    Provide additional training examples to a subject model, in order to improve the accuracy on the
    specific examples

    WARNING: This function is currently broken. Do not use until it has been debuged!!!

    :type subject_model: SubjectOppenheimSigmoid
    :param subject__model: The subject neurual tet to train

    :type train_set: Tuple[,]
    :param train_set: A pair of feature matrix and target label vector

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :rtype: None
    """

    (train_x, train_y) = train_set

    print(train_x, ' ', train_y)

    # compute number of minibatches for training, validation and testing
    #    n_train_batches = train_x.shape[0] // batch_size
    n_train_batches = 1
    batch_size = train_x.shape[0]

    # make datasets compatible with Theano
    train_set_x = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX),
                                borrow=True)
    # train_set_y_temp = theano.shared(numpy.asarray(train_y, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y, dtype='int32'), borrow=True)
    # train_set_y = T.cast(train_set_y_temp, 'int32')
    # train_set_y = train_set_y_temp
    print(train_set_x, '  ', train_set_y)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = subject_model.loss_function(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=subject_model.W)
    g_b = T.grad(cost=cost, wrt=subject_model.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(subject_model.W, subject_model.W - learning_rate * g_W),
               (subject_model.b, subject_model.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    # train_model = theano.function(
    #     inputs=[index],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    train_model = theano.function(
        inputs=[subject_model.input, y],
        outputs=cost,
        updates=updates
    )


    ###############
    # TRAIN MODEL #
    ###############
    # considered significant

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        epoch_cost = 0;
        for minibatch_index in range(n_train_batches):
            batch_cost = train_model(train_set_x, train_set_y)
            epoch_cost =  epoch_cost + batch_cost
            # iter = (epoch - 1) * n_train_batches + minibatch_index

        # periodically report the cost (every 5 epochs)
        if (epoch % 1) == 0:
            print(
                (
                    '     epoch %i, cost of  %4.1f%%'
                ) %
                (
                    epoch, epoch_cost
                )
            )



def recall_as_learning(subject_model, target_id, concept_vec, learning_rate=0.2, verbose = False):
    """ Present a single concept to the subject (one trial), and adjust the model weights
        in order to simulate the priming effect.

    :type subject_model: SubjectOppenheimSigmoid
    :param subject_model: The pre-trained neural network representing the subject
    :param target_id: The integer id for the concept the subject is presented with (and is supposed to recall)
    :param concept_vec: A vector with a separate column set to 1 for each feature present. Note, we are assuming that
        a single concept is presented here.
    :return:
    """

    # to-do: this whole process calculates the outputs of the NNet three times using the same input activations!!!
    # can we make it more efficient?

    naming_error = False
    timeout = False

    # compile a predictor function
    # this function requires an array-like object as input
    predict_model = theano.function(
        inputs=[subject_model.input],
        outputs=subject_model.y_pred)
    # compute the time for this prediction
    # this must occur before we adjust the weights, in order to accurately reflect the recall process
    time = selection_time(subject_model, concept_vec, verbose)

    # if the data is a theano variable, this is how to get the array/matrix...
    # input_x = test_set_x.get_value()

    # whatever is recalled, will be treated as the target for learning (thus priming it)
    predict_y = predict_model([concept_vec])
    predict_y = predict_y.astype(numpy.int32)  # convert from int64 to int32 for use later...

    # declare a variable to be used in the cost and train_model functions
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = subject_model.loss_function(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=subject_model.W)
    g_b = T.grad(cost=cost, wrt=subject_model.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(subject_model.W, subject_model.W - learning_rate * g_W),
               (subject_model.b, subject_model.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[subject_model.input, y],
        outputs=cost,
        updates=updates
    )

    if verbose is False:
        # right now the outputs are just cost, but it would probably be more useful to make it y_pred....
        # results = train_model([concept_vec], predict_y)     # note, any change to this line must be replicated in the else
        # this is the same when subject is correct, but will gradually improve if incorrect
        results = train_model([concept_vec], [target_id])     # note, any change to this line must be replicated in the else
    else:
        W_old = subject_model.W.get_value()
        # results = train_model([concept_vec], predict_y)     # change this line if you change the one above!!!!
        results = train_model([concept_vec], [target_id])     # change this line if you change the one above!!!!
        W_new = subject_model.W.get_value()
        W_delta = W_new - W_old
        print("Nonzeros: ", numpy.count_nonzero(W_delta))
        hist, bins = numpy.histogram(W_delta, bins=(-1,-0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1))
        print("Histogram: ", hist, ", Bins:", bins)

    # note, predict_y will be a vector, since it is possible to make predictions for many examples simultaneously
    if predict_y[0] != target_id:
        naming_error = True
        if verbose:
            print('Naming error: ', predict_y[0], ' instead of ', target_id)
    # have a time out that essentially cuts off the timings
    if time > deadline:
        time = deadline
        timeout = True

    return time, naming_error, timeout

