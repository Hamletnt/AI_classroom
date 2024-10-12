from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

df = pd.read_csv('cheese_encode_4_fatcal_clean_input.csv')

# Function to fill NaN in the sample data based on nearest neighbors
def fill_missing_values(sample_data, df):
    # Get indices where there are no NaN values (only known values)
    known_columns = ~np.isnan(sample_data)

    # Extract only known values
    known_values = sample_data[known_columns]

    # Select the corresponding columns from the dataset that match known values
    columns_to_use = np.where(known_columns)[0]

    # Extract the relevant columns from the dataset for comparison
    df_relevant = df.iloc[:, columns_to_use]

    # Initialize the Nearest Neighbors model
    nearest_neighbors = NearestNeighbors(n_neighbors=1)

    # Fit the model on the relevant columns of the dataset
    nearest_neighbors.fit(df_relevant)

    # Find the closest row in the dataset to the known values
    _, indices = nearest_neighbors.kneighbors([known_values])

    # Get the closest row from the original dataset
    closest_row = df.iloc[indices[0][0]].values

    # Fill in the NaN values in the sample data with the closest row's values
    filled_sample = sample_data.copy()

    # Replace NaN values with the corresponding values from the closest row
    nan_columns = np.isnan(filled_sample)
    filled_sample[nan_columns] = closest_row[nan_columns]

    return filled_sample

# Decision Tree Node Class (Unchanged)
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value

# Decision Tree Classifier (Unchanged)
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    def build_tree(self, dataset, curr_depth=0):
        ''' Recursive function to build the tree ''' 
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
        
            if best_split and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                      left_subtree, right_subtree, best_split["info_gain"])

        # Create a leaf node if no further split is possible
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' Function to find the best split '''
    
        # Dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
    
        # Loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
        
            # Loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # Get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
            
                # Check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                
                    # Compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                
                    # Update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain

        # Return best split if found, otherwise return an empty dictionary
        return best_split if "info_gain" in best_split else None
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="gini"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Random Forest Classifier Implementation
class RandomForestClassifierFromScratch:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, Y_sample = self._bootstrap_sampling(X, Y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)

    def _bootstrap_sampling(self, X, Y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], Y[indices]

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = [self._majority_vote(tree_predictions[:, i]) for i in range(X.shape[0])]
        return final_predictions

    def _majority_vote(self, predictions):
        return max(set(predictions), key=list(predictions).count)

# โหลดโมเดลที่ได้รับการฝึกมา
loaded_rf_classifier = joblib.load('model.joblib')

class_names = [
    'Blue',          # 0
    'Brie',          # 1
    'Camembert',     # 2
    'Cheddar',       # 3
    'Cottage',       # 4
    'Feta',          # 5
    'Gouda',         # 6
    'Mozzarella',    # 7
    'Parmesan',      # 8
    'Pasta filata',  # 9
    'Swiss Cheese',  # 10
    'Tomme',         # 11
    'Pecorino'       # 12
]

predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Get JSON data from the request
        input_data = np.array(data['input'], dtype=np.float64)  # Convert input to a numpy array with float64 type

        # Ensure the input data is 2D (even if it's a single sample)
        if input_data.ndim == 1:
            input_data = np.expand_dims(input_data, axis=0)

        # Convert None (or null in JSON) values to np.nan
        input_data = np.where(input_data == None, np.nan, input_data)

        # Fill missing values (NaNs) using nearest neighbors
        filled_input_data = np.array([fill_missing_values(row, df) for row in input_data])

        # Predict using the loaded random forest classifier
        predictions = loaded_rf_classifier.predict(filled_input_data)

        # Convert predictions to a list of integers (JSON-serializable)
        predictions = [int(pred) for pred in predictions]

        # Convert predictions to their corresponding cheese class names
        predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]


        return jsonify({'prediction': predictions+predicted_class_names})  # Return the predictions as a JSON response
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return the error message as a JSON response

@app.route('/')
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # เปิดให้เข้าถึงจากทุกอินเทอร์เฟซที่พอร์ต 5000
