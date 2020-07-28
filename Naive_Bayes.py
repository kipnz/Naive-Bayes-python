import csv
import math


def cross_validation_splitting(mydict, fold):
    #this function splits the data into sections of however many folds the user specifies 
    split = {}
    fold_size = int(len(mydict) / fold)

    for i in range(1, fold+1):
        split[i] = {}
        start = (i-1)*fold_size

        for j in range(1, fold_size+1):
            step = j+ start
            split[i][j] = mydict[step]

    return split


def create_class_word_counts(mydict):
    #this section of code creates a new dictionary that gives the classes,
    #all the words in the class, and their counts in that class
    #the dictionary getting put in as a dictionary of dictionaries {row : {class : attribute string}...}
    word_counts = {}
    for row in mydict:
        class_row = list(mydict[row].keys())[0]

        if class_row not in word_counts:
            temp = {}

            for word in mydict[row][class_row].split():

                if word not in temp:
                    temp[word] = 1
                else:
                    temp[word] += 1
                    
            word_counts[class_row] = temp

        else:

            for word in mydict[row][class_row].split():

                if word not in word_counts[class_row]:
                    word_counts[class_row][word] = 1
                else:
                    word_counts[class_row][word] += 1


    #here uses inverse_document_freq to change the counts weightings
    inverse_dict = inverse_document_freq(mydict)
    
    for cls in word_counts:
        for word in word_counts[cls]:
            word_counts[cls][word] = word_counts[cls][word] * inverse_dict[word]

    return word_counts

def inverse_document_freq(base_dict):
    #this function implements the inverse document frequency extension to the Naive Bayes algorithms
    #it takes the dictionary in form {row : {class : abstract...}... } and gives a dictionary of every word in the document as the key
    #and the item is the log(total rows / total rows a word is in) part of the inverse document frequency equation 
    total = {}
    
    for row in base_dict:
        class_row = list(base_dict[row].keys())[0]
        temp = {}
        for word in base_dict[row][class_row].split():
            if word not in temp:
                temp[word] = 1
        for word in temp:
            if word not in total:
                total[word] = 1
            else:
                total[word] += 1

    inverse_dict = {}
    for word in total:
        inverse_dict[word] = math.log(len(base_dict) / total[word])

    return inverse_dict




def run_cross_validation(mydict, split, top_num):
    #this is given the dictionary of all rows no changes
    #split is the cross validation split
    #top_num is the amount of words we want to consider
    
    split_dict = cross_validation_splitting(mydict,split)
    total = 0
    
    for element in split_dict:
        train_dict = {}
        test_dict = {}
        test_dict.update(split_dict[element])
        count = 1
        #this series of loops combines the training data into one concurrent dictionary

        for dictionary in split_dict:
            if dictionary != element:
                for sub_dict in  split_dict[dictionary]:
                    train_dict[count] = split_dict[dictionary][sub_dict]
                    count += 1

        #this code takes the traing set and processes it getting class counts of each word, total counts of words, the top however many words
        class_counts = create_class_word_counts(train_dict)
        total_dict = get_total_word_counts(class_counts)
        top_words = get_top_words(total_dict, top_num)
        prior = get_priors(train_dict)
        conditional = get_conditional_prob(top_words, class_counts)
        compare_dict = classify(test_dict, conditional, prior)

        clean_test = validation_test_classes(test_dict)
        
        inn = 0 
        for element in compare_dict:
            if compare_dict[element] == clean_test[element]:
                inn += 1
        total += (inn / len(compare_dict))
        
    return (total / split)



def validation_test_classes(test_set):
    #this function cleans the test set to only be the row and the abstract data
    clean = {}

    for row in test_set:
        class_row = list(test_set[row].keys())[0]
        clean[row] = class_row
    return clean
        


def classify(test_set, conditional_prob, prior_prob):
    #test_set = {row : {class (empty with new test data): abstract)}}
    #conditional probability = {word : {class : probability ... } ... }
    #prior_prob = {class: proabaility...}
    #the two prbability dictionaries are in alphabetical order
    #this code classifies the data by going row by row, class by class and word by word,
    #multiplying the probabilities and adding them to a temporary dictionary of the probabilities of each class
    #we find the max of these probabilties and then classify that piece of data in a new dictionary with just the row number and the classification
    class_dict = {}
    clean_test_set = test_set_word_count(test_set)

    for row in clean_test_set:
        prob_dict = {}

        for clas in prior_prob:
            prob = 1.0

            for word in clean_test_set[row]:

                if word not in conditional_prob:
                    continue
                else:
                    prob = prob + (conditional_prob[word][clas] * clean_test_set[row][word])

            prob = prob + prior_prob[clas]
            prob_dict[clas] = prob

        class_dict[row] = max(prob_dict, key = prob_dict.get)
        
    return class_dict


def test_set_word_count(test_set):
    clean_test_set = {}

    for row in test_set:
        temp = {}
        class_row = list(test_set[row].keys())[0]

        for word in test_set[row][class_row].split():

            if word not in temp:
                temp[word] = 1
            else:
                temp[word] += 1

        clean_test_set[row] = temp
    return clean_test_set


    
def get_conditional_prob(top_words, class_counts):
    #this function creats a dictionary in the style {keyout : {keyin: conditional probability ...} ... } where
    #the key out is on of the most common words within a range specified by the user in get_top_words()
    # the keyin is the classes (in alphabetical order) that the conditional probability refers to.
    conditional_prob_dict = {}
    total_words_in_class = {}
    total_words = len(top_words)
    for clas in class_counts:
        count = 0

        for word in class_counts[clas]:

            if word in top_words:
                count += class_counts[clas][word]

        total_words_in_class[clas] = count

    
    for word in top_words:
        conditional_prob_dict[word] = {}

        for cls in class_counts:
            if word not in class_counts[cls]:
                conditional_prob_dict[word][cls] = math.log((1) / (total_words_in_class[cls] + total_words))
            else:
                conditional_prob_dict[word][cls] = math.log((1 + class_counts[cls][word]) / (total_words_in_class[cls] + total_words))

        conditional_prob_dict[word] = dict(sorted(conditional_prob_dict[word].items()))
    return conditional_prob_dict


def get_priors(data_dict):
    #this function gets the training set that is a dictionary of each row (no counting has been done) e.g. {row: {class: abstract ... } ... }
    #and computes the priors
    prior_dict = {}
    for element in data_dict:
        cls = list(data_dict[element].keys())[0]
        if cls not in prior_dict:
            prior_dict[cls] = 1
        else:
            prior_dict[cls] += 1
    for clas in prior_dict:
        prior_dict[clas] = math.log(prior_dict[clas] / float(len(prior_dict)))
    
    return dict(sorted(prior_dict.items()))
            
    
def get_total_word_counts(mydict):
    #this gives an alphabetically ordered dictionary of the words total counts
    word_total = {}
    for element in mydict:
        for word in mydict[element]:
            if word not in word_total:
                word_total[word] = mydict[element][word]
            else:
                word_total[word] += mydict[element][word]
    word_total = dict(sorted(word_total.items()))
    return word_total

def get_top_words(mydict, num_words):
    #we assume we have an alphabetically ordered dictionary here from word counts in order for this to work
    top_words = {}
    tally_dict = {}
    short_tally = {}
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                  'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                  'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                  'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                  'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
                  'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                  'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                  "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    #here I created tallys of all total counts and creates an ordered dictionary from largest count to smallest
    #it also doesn't count stop words
    for element in mydict:
        if element not in stop_words:
            if mydict[element] not in tally_dict:
                tally_dict[mydict[element]] = 1
            else:
                tally_dict[mydict[element]] += 1
    tally_dict = dict(reversed(sorted(tally_dict.items())))
    count = 0

    #here i go through the dictionary, in order, adding tally's until i reach the amount equal to the num_words variable.
    #if we reach a tally that goes beyond num_words, this code just reduces the number in order to be equal to num_words.
    for element in tally_dict:
        if count < num_words:
            if count + tally_dict[element] > num_words:
                short_tally[element] = tally_dict[element] - (count + tally_dict[element] - num_words)
                break
            else:
                count += tally_dict[element]
                short_tally[element] = tally_dict[element]

    #and then using that dictionary of shortened tally's i itterate through every word in order and if it is in the tally and there is enough room for it it will
    #be added to the top words in the classification
    for element in mydict:
        if mydict[element] in short_tally and short_tally[mydict[element]] != 0:
            top_words[element] = mydict[element]
            short_tally[mydict[element]] -= 1
    
    return top_words
          

def main():

    dicti = {}

    #this will convert the file into a dictionary of each instance with a dictionary
    #of its classification and the abstract string
    print("Importing test set...")
    with open('trg.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()
        dicti = {int(rows[0]):{rows[1]:rows[2]} for rows in reader}

    #this code will make a call to run 10 fold cross validation on the test set and return its average accuracy
    print("running cross validation...")
    percentage = run_cross_validation(dicti, 10, 40000)
    print("accuracy percentage: {:.3f}".format(percentage * 100))


    #this code runs the full unchanged test set through functions to give first, a count of how many times a word appears in a class
    #second it will get the total count of every word
    #it will then get the top 10,000 words based on their counts
    #and then get the prior probabilities and the conditional probabilities
    print("learning from full test set...")
    class_counts = create_class_word_counts(dicti)
    total_dict = get_total_word_counts(class_counts)
    top_words = get_top_words(total_dict, 40000)
    prior = get_priors(dicti)
    conditional = get_conditional_prob(top_words, class_counts)

    test = {}

    #this code will change the test set to a dictionary that we can use, N is a placeholder class that doesnt exist
    print("processing test set...")
    with open('tst.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()
        test = {int(rows[0]):{"N":rows[1]} for rows in reader}

    #the test set is then classified with this function 
    print("classifying test set...")
    classified = classify(test, conditional, prior)


    #the classified test set is then written to a csv file 
    with open('Classifications.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'class'])
        for row in classified:
            writer.writerow([row, classified[row]])

    print('Finished!')


main()
