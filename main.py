#Project 1
#Farhat Lamia Barsha

########################################################################################################################
#Part 1 - Dataset preparation
import numpy as np
import pandas as pd
from scipy.io import arff
import sys

#Read the training arff file to get attribute names
train_data = arff.loadarff('/Users/farhatlamiabarsha/Downloads/DecisionTree/KDDTrain+.arff')
df = pd.DataFrame(train_data[0])
# print(df)

#getting attribute names
attribute_names = df.columns.tolist()
# print(attribute_names)

#reading txt file of train and test dataset
df_train = pd.read_csv("/Users/farhatlamiabarsha/Downloads/DecisionTree/KDDTrain+.txt", header=None)
df_test = pd.read_csv("/Users/farhatlamiabarsha/Downloads/DecisionTree/KDDTest+.txt", header=None)
# print(df_train.columns)

#dictionary to map old column names to new ones
column_mapping = {
    0: 'duration',
    1: 'protocol_type',
    2: 'service',
    3: 'flag',
    4: 'src_bytes',
    5: 'dst_bytes',
    6: 'land',
    7: 'wrong_fragment',
    8: 'urgent',
    9: 'hot',
    10: 'num_failed_logins',
    11: 'logged_in',
    12: 'num_compromised',
    13: 'root_shell',
    14: 'su_attempted',
    15: 'num_root',
    16: 'num_file_creations',
    17: 'num_shells',
    18: 'num_access_files',
    19: 'num_outbound_cmds',
    20: 'is_host_login',
    21: 'is_guest_login',
    22: 'count',
    23: 'srv_count',
    24: 'serror_rate',
    25: 'srv_serror_rate',
    26: 'rerror_rate',
    27: 'srv_rerror_rate',
    28: 'same_srv_rate',
    29: 'diff_srv_rate',
    30: 'srv_diff_host_rate',
    31: 'dst_host_count',
    32: 'dst_host_srv_count',
    33: 'dst_host_same_srv_rate',
    34: 'dst_host_diff_srv_rate',
    35: 'dst_host_same_src_port_rate',
    36: 'dst_host_srv_diff_host_rate',
    37: 'dst_host_serror_rate',
    38: 'dst_host_srv_serror_rate',
    39: 'dst_host_rerror_rate',
    40: 'dst_host_srv_rerror_rate',
    41: 'class'
}

df_train.rename(columns=column_mapping, inplace=True)
df_test.rename(columns=column_mapping, inplace=True)

#removing the last column 42
df_train = df_train.iloc[:, :-1]
df_test = df_test.iloc[:, :-1]
# print(df_train)

########################################################################################################################
#Part 2 algorithm implementation
#decision tree learning algorithm
def decision_tree_learning(examples, attributes, default=None):
    if len(examples) == 0: #if there are no nodes more like leaf
        return default  #return the default value
    elif all_same_classification(examples):  #if all the examples have the same classification
        return examples['class'].iloc[0]  #return the class label of that single class.
    elif len(attributes) == 0:  #if there are no more attributes left to split on
        return majority_value(examples)  #return the most frequent class label among the remaining examples.
    else:
        best_attribute = argmax_importance(attributes, examples)  #to find the best attribute
        tree = {'attribute': best_attribute, 'children': {}} #initialization

        #loop iterates over unique values of the best_attribute in the current set of examples
        for value in examples[best_attribute].unique():
            exs = examples[examples[best_attribute] == value]
            subtree = decision_tree_learning(exs, [attr for attr in attributes if attr != best_attribute], examples)
            tree['children'][value] = subtree

        return tree

#checks if all examples have same classification
def all_same_classification(examples):
    return len(set(np.array(examples)[:, -1])) == 1 #check the number of unique classification labels in the dataset

#calculates the majority class and return it
def majority_value(examples):
    values, counts = np.unique(examples['class'], return_counts=True) #values hold unique class label, counts holds that label's occurence
    return values[np.argmax(counts)] #argmax function return the maxium occured class label

#selects the best attribute by calculating information gain
def argmax_importance(attributes, examples):
    best_attribute = None    #initilization
    best_information_gain = -1

    #calculates total entropy of the dataset
    total_entropy = calculate_entropy(examples)
    # iterates through attribute and extract unique values
    for attribute in attributes:
        attribute_values = examples[attribute].unique()
        weighted_entropy = 0.0

        #for each unique value of current attribute create subset and calculate weight and weighted entropy
        for value in attribute_values:
            subset = examples[examples[attribute] == value]
            subset_weight = len(subset) / len(examples)
            weighted_entropy += subset_weight * calculate_entropy(subset)
        information_gain = total_entropy - weighted_entropy

        #update best attribute according to information gain
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_attribute = attribute

    return best_attribute

#calculated the entropy of the dataset
def calculate_entropy(examples):
    total_examples = len(examples)
    #if dataset is empty
    if total_examples == 0:
        return 0.0

    #counts the occurrences of each unique class label
    class_counts = examples['class'].value_counts()
    entropy = 0.0

    #this loop iterates through the counts of each unique class label
    for class_count in class_counts:
        class_probability = class_count / total_examples
        entropy -= class_probability * np.log2(class_probability)

    return entropy

#recursively traverses the tree and prints the nodes in a preorder fashion
def preorder_traverse(tree, depth=0):
    if isinstance(tree, dict): #if the tree is dictionary
        for children, subtree in tree['children'].items():  #iterates over the children
            print(' ' * depth, children)
            preorder_traverse(subtree, depth + 1) #recursively calls itself to traverse the child node's subtree
    else:
        print(' ' * depth, tree)


########################################################################################################################
#Part 3 Generating decision tree
#define the list of attributes
attributes = list(df_train.columns[:-1])
attributes_test = attributes

#decision tree building
decision_tree = decision_tree_learning(df_train, attributes)

def predict(instance, tree):
    if isinstance(tree, dict): #if tree is a dictionary
        attribute = tree['attribute'] #extract attribute name
        if attribute in instance: #if attribute exists in instance
            value = instance[attribute] #extract the value associated with attribute name
            if value in tree['children']: #if the extracted value is child
                subtree = tree['children'][value] #extract the corresponding subtree
                return predict(instance, subtree) #make a recursive call with subtree
        return None
    else:
        return tree

#prediction for training data
predicted_train_labels = []
for index, train_row in df_train.iterrows(): #iterates through each row of training set
    train_instance = train_row.drop('class').to_dict() #create a dictionary
    predicted_label = predict(train_instance, decision_tree) #do prediction
    predicted_train_labels.append(predicted_label)

#prediction for the test data same as training data
predicted_test_labels = []
for index, test_row in df_test.iterrows():
    test_instance = test_row.drop('class').to_dict()
    predicted_label = predict(test_instance, decision_tree)
    predicted_test_labels.append(predicted_label)

#convert predicted labels list to pandas Series
predicted_train_labels_series = pd.Series(predicted_train_labels)
predicted_test_labels_series = pd.Series(predicted_test_labels)

#adding predicted labels series in the training and testing set
df_train['predicted_class'] = predicted_train_labels_series
df_test['predicted_class'] = predicted_test_labels_series

#accuracy for the training data
train_accuracy = (df_train['class'] == df_train['predicted_class']).mean()
#print("Training Accuracy:", train_accuracy)
train_accuracy_percentage = train_accuracy * 100
print("Training Accuracy:", train_accuracy_percentage, "%")

#accuracy for the test data
test_accuracy = (df_test['class'] == df_test['predicted_class']).mean()
#print("Test Accuracy:", test_accuracy)
test_accuracy_percentage = test_accuracy * 100
print("Test Accuracy:", test_accuracy_percentage, "%")

#decision tree print function
def print_decision_tree(tree, depth=0):
    if isinstance(tree, dict):
        attribute_name = tree['attribute']
        print(' ' * depth, "Attribute:", attribute_name)

        for value, subtree in tree['children'].items():
            print(' ' * depth, "Value:", value)
            print_decision_tree(subtree, depth + 2)
    else:
        print(' ' * depth, "Class:", tree)

print("Decision Tree:")
print_decision_tree(decision_tree)

#output to txt file
output_file_path = '/Users/farhatlamiabarsha/Downloads/DecisionTree/output.txt'
with open(output_file_path, 'w') as output_file:
    sys.stdout = output_file
    print("Training Accuracy:", train_accuracy_percentage, "%")
    print("Test Accuracy:", test_accuracy_percentage, "%")
    print("Decision Tree:")
    print_decision_tree(decision_tree)
sys.stdout = sys.__stdout__
print(f"Output has been written to {output_file_path}")






