import numpy as np

class TinyDecisionTree:

    def __init__(self):
        '''
        Initialize the tree structure.

        Define:
        - self.feature_idx: index of the feature used for splitting
        - self.threshold: threshold value to split the feature
        - self.left_class: predicted class if feature value <= threshold
        - self.right_class: predicted class if feature value > threshold
        '''

        self.feature_idx = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        '''
        Fit the tree using training data.

        Instruction 
