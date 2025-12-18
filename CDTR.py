import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CustomDecisionTreeRegressor:

    def __init__(self, max_depth, min_samples_leafs):
        self.max_depth = max_depth
        self.min_samples_leafs = min_samples_leafs
        self.tree_=None

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_rss = float("inf")

        for f in range(n_features):
            # Extract feature + sort once
            feature_values = X[:, f]
            sorted_idx = np.argsort(feature_values)
            fv_sorted = feature_values[sorted_idx]
            y_sorted = y[sorted_idx]

            # Precompute global sums (for right side initially)
            total_count = n_samples
            total_sum = np.sum(y_sorted)
            total_sq_sum = np.sum(y_sorted ** 2)

            # Cumulative sums for left region (initially empty)
            left_count = 0
            left_sum = 0.0
            left_sq_sum = 0.0

            for i in range(n_samples - 1):
                yi = y_sorted[i]

                # Move one sample from right → left
                left_count += 1
                left_sum += yi
                left_sq_sum += yi * yi

                # Right side = global - left
                right_count = total_count - left_count
                if right_count == 0:
                    break

                right_sum = total_sum - left_sum
                right_sq_sum = total_sq_sum - left_sq_sum

                # Skip if feature value doesn't change → no valid threshold
                if fv_sorted[i] == fv_sorted[i + 1]:
                    continue

                # Compute RSS for left and right using variance formula
                rss_left = left_sq_sum - (left_sum ** 2) / left_count
                rss_right = right_sq_sum - (right_sum ** 2) / right_count
                total_rss = rss_left + rss_right

                if total_rss < best_rss:
                    best_rss = total_rss
                    best_feature = f
                    best_threshold = (fv_sorted[i] + fv_sorted[i + 1]) / 2

        return best_feature, best_threshold
    
    def fit(self, X,y):
        self.tree_ = self.build_tree(X, y, depth=0)
        return self

    # Recursive Tree Builder 
    def build_tree(self, X, y, depth=0):
        
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_leafs:
            return Node(value=np.mean(y))

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Node(value=np.mean(y))

        left_idx = X[:, feature] < threshold
        right_idx = X[:, feature] >= threshold

        left = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right = self.build_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    # Prediction 
    def predict_one(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self.predict_one(node.left, x)
        else:
            return self.predict_one(node.right, x)

    def predict(self, X):
        return np.array([self.predict_one(self.tree_, x) for x in X])