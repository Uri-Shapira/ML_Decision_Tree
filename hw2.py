import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if(len(data) > 0):
        rows = len(data)
        sum_of_ones = sum(data[:,-1])
        proportion_of_ones = sum_of_ones/rows 
        proportion_of_zeros = 1- proportion_of_ones
        gini = (proportion_of_ones**2 + proportion_of_zeros**2)
    else:
        return 0
                
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (1-gini)

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if(len(data) == 0):
        return 0
    else:
        rows = len(data[:,-1])
        sum_of_ones = sum(data[:,-1])
        proportion_of_ones = sum_of_ones/rows 
        if proportion_of_ones == 0:
            log1 = 0
        else:
            log1 = np.log(proportion_of_ones)
            
        proportion_of_zeros = 1- proportion_of_ones
        
        if proportion_of_zeros == 0:
            log0 = 0
        else:
            log0 = np.log(proportion_of_zeros)
        
        entropy = proportion_of_ones * log1 +  proportion_of_zeros * log0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return - entropy

def split_attribute(feature ,value, data):
    """
    Split an attribute column to two "children columns" 
    """

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    a = data[data[:, feature] < value]
    b = data[data[:, feature] >= value]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return a,b

def get_average_values(data):
    """
    Create a matrix of average values to test for the optimal impurity split. 
    """
    split_values = [[] for i in range(np.size(data, 1) - 1)]
    for i in range(np.size(data, 1) - 1):
        current_datasort = data[data[:, i].argsort()]
        for j in range(np.size(data, 0) - 1):
            average = (current_datasort[j, i] + current_datasort[j + 1, i]) / 2 
            split_values[i].append(average)

    return np.array(split_values)

def impurity_gain(data, attribute, split_value, impurity):
    """
    splits an attribute into two sets and checks the impurity gain of such a split
    """
    a,b = split_attribute(attribute, split_value, data)
    impurity_a = len(a)/(len(a) + len(b)) * impurity(a)
    impurity_b = len(b)/(len(a) + len(b)) * impurity(b)
    gain = impurity(data) - (impurity_a + impurity_b)

    return a, b, gain

def find_best_split(data , impurity):
    """
    Finds the optimal attribute and value to split on for a given data set 
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    attribute, split_value, best_gain = 0, 0, 0
    less_than_child = np.array([])
    greater_than_child = np.array([])
    split_values = get_average_values(data)
    for i in range(np.size(split_values, 0)):
        for j in range(np.size(split_values, 1)):
            a, b, gain = impurity_gain(data, i, split_values[i,j], impurity)
            if gain > best_gain:
                attribute, split_value, best_gain, less_than_child, greater_than_child = i, split_values[i,j], gain, a ,b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return attribute, split_value, less_than_child, greater_than_child


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []

    def add_child(self, node):
        self.children.append(node)
    
    def add_parent(self, node):
        self.parent = node
    
def get_value(node):
    if sum(node.data[:,-1])*2 >= len(node.data):
        return 1
    else:
        return 0

def build_tree(data, impurity):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
       as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    leaves = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(None,None)
    root.data = data
    build_tree_2(root, impurity, leaves)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root, leaves

def build_tree_2(node, impurity, leaves):
    if impurity(node.data) == 0:
        leaves.append(node)
        return node
    feature, split_value, a, b = find_best_split(node.data, impurity)
    node.value = split_value
    node.feature = feature
    node_a = DecisionNode(None,None)
    node_a.data = a
    node_b = DecisionNode(None,None)
    node_b.data = b
    node.add_child(node_a)
    node.add_child(node_b)
    node_a.add_parent(node)
    node_b.add_parent(node)
    build_tree_2(node_a, impurity, leaves)
    build_tree_2(node_b, impurity, leaves)
    return node


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    while node.children != []:
        if instance[node.feature] < node.value:
            node = node.children[0]
        else:
            node = node.children[1]         
    ratio_of_ones = sum(node.data[:,-1]) / len(node.data)
    if ratio_of_ones > 0.5:
       pred = 1
    else:
        pred = 0
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    for row in dataset:
        pred = predict(node, row)
        if pred == row[-1]:
            accuracy+=1
    accuracy = accuracy/len(dataset)
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def remove_children_from_tree(parent):
    parent.children = []
            
def return_children_to_tree(tempTree):
    tempTree[0].add_child(tempTree[1])
    tempTree[0].add_child(tempTree[2])

def count_nodes(node):
    if node is None:
        return 0
    if node.children == []:
        return 1
    return 1 + count_nodes(node.children[0]) + count_nodes(node.children[1])  
     
def prune_tree(X_train, accuracy_data):
    internal_nodes = []
    accuracy = []
    tree, leaves = build_tree(X_train, calc_entropy)
    accuracy.append(calc_accuracy(tree, accuracy_data))
    internal_nodes.append(count_nodes(tree) - len(leaves))
    i = 0
    while(tree.children != []):
        tree, leaves, accuracy = trim_parent(tree, leaves, accuracy, accuracy_data, internal_nodes)
    return accuracy, internal_nodes

def trim_parent(tree, leaves, accuracy, accuracy_data, internal_nodes):
    index = np.random.randint(0,len(leaves)-1)
    parent_to_remove = leaves[index]
    tempTree = [leaves[index].parent, leaves[index].parent.children[0], leaves[index].parent.children[1]]
    remove_children_from_tree(leaves[index].parent)
    curr_accuracy = calc_accuracy(tree, accuracy_data)
    return_children_to_tree(tempTree)
    for i in range(1 , len(leaves) - 1):
        if(i != index):
            tempTree = [leaves[i].parent, leaves[i].parent.children[0], leaves[i].parent.children[1]]
            remove_children_from_tree(leaves[i].parent)
            new_accuracy = calc_accuracy(tree, accuracy_data)
            return_children_to_tree(tempTree)
            if(new_accuracy > curr_accuracy):
                curr_accuracy = new_accuracy
                parent_to_remove = leaves[i]
    accuracy.append(curr_accuracy)
    delete_subtree(parent_to_remove.parent, leaves)
    leaves.append(parent_to_remove.parent)
    internal_nodes.append(count_nodes(tree) - len(leaves))
    return tree, leaves, accuracy
        
def delete_subtree(parent, leaves):
    if(parent.children != []):
        delete_subtree2(parent.children[0], leaves)
        delete_subtree2(parent.children[1], leaves)
    parent.children = []

def delete_subtree2(child, leaves):
    if(child.children != []):
        delete_subtree2(child.children[0], leaves)
        delete_subtree2(child.children[1], leaves)
    if child in leaves:
        leaves.remove(child)


def print_tree(node, tab):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''
                               
    if(node.children == []):
        value = get_value(node)
        print("\t"*tab, "Leaf: [{"+str(value)+":", str(len(node.data))+"}]")
    else:                                                                 
        print("\t"*tab, "[X"+str(node.feature), "<=", str(node.value)+"]")
        tab += 1  
        print_tree(node.children[0], tab)
        print_tree(node.children[1], tab)
    
    
    ###########################################################################    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
