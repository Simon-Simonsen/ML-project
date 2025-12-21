import numpy as np

def rss(y):
    return len(y) * np.var(y) if len(y) > 0 else 0

def proportions(region):
    """Return class 0 and 1 proportions in a region"""
    if len(region) == 0:
        return (0, 0)
    count0 = sum(1 for _, ci in region if ci == 0)
    count1 = sum(1 for _, ci in region if ci == 1)
    total = count0 + count1
    return (count0 / total, count1 / total)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def best_split(X, y):
    n_samples, n_features = X.shape
    best_feature, best_threshold = None, None
    best_rss = float("inf")

    for feature in range(n_features):
        values = X[:, feature]
        thresholds = np.unique(values)
        for t in thresholds:
            left_idx = values < t
            right_idx = values >= t
            if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                continue

            rss_left = rss(y[left_idx])
            rss_right = rss(y[right_idx])
            total_rss = rss_left + rss_right

            if total_rss < best_rss:
                best_rss = total_rss
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

# --- Recursive Tree Builder ---
def build_tree(X, y, depth=0, max_depth=8, min_leaves = 5):
    
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < min_leaves:
        return Node(value=np.mean(y))

    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=np.mean(y))

    left_idx = X[:, feature] < threshold
    right_idx = X[:, feature] >= threshold

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    return Node(feature, threshold, left, right)

# --- Prediction ---
def predict_one(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(node, X):
    return np.array([predict_one(node, x) for x in X])