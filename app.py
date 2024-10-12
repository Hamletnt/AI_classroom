from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import encoder as en
from fuzzywuzzy import process
from flask_cors import CORS

app = Flask(__name__)


# เปิดใช้งาน CORS สำหรับทุกเส้นทางในแอปพลิเคชัน (สามารถลบ CORS(app) นี้ได้ถ้าใช้ฟังก์ชันด้านบน)
CORS(app, origins="http://127.0.0.1:5500") 

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

# Helper function to get the closest match using fuzzy matching
def get_closest_match(value, valid_values):
    if isinstance(value, str):  # Check if the value is a string before doing fuzzy matching
        match, score = process.extractOne(value, valid_values)
        return match if score >= 70 else None  # Only return match if score >= 70
    return None  # If the value is not a string, return None

# @app.route('/predict', methods=['POST', 'OPTIONS'])
# def predict():
#     if request.method == 'OPTIONS':
#         # Handle the preflight request
#         response = app.make_default_options_response()
#         headers = response.headers
#         headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
#         headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
#         headers['Access-Control-Allow-Headers'] = 'Content-Type'
#         return response, 200

#     try:
#         # Handle the actual POST request
#         data = request.get_json(force=True)  # Get JSON data from the request
#         input_data = data['input']  # Expecting a list of strings or nulls

#         # Convert input values using static_mapping and fuzzy matching
#         encoded_input = []
#         for col_name, value in zip(en.static_mapping.keys(), input_data[0]):
#             if value is None:  # If the value is null, keep it as is
#                 encoded_input.append(None)
#             elif value in en.static_mapping[col_name].keys():  # Exact match
#                 encoded_input.append(en.static_mapping[col_name][value])
#             else:  # Use fuzzy matching to find the closest value
#                 closest_match = get_closest_match(value, en.static_mapping[col_name].keys())
#                 if closest_match:
#                     encoded_input.append(en.static_mapping[col_name][closest_match])
#                 else:
#                     return jsonify({'error': f'No match found for {value} in {col_name}'}), 400

#         encoded_input = np.array(encoded_input).reshape(1, -1)  # Ensure input is 2D for prediction

#         # Predict using the loaded random forest classifier
#         predictions = loaded_rf_classifier.predict(encoded_input)

#         # Convert predictions to class names
#         class_names = ['Blue', 'Brie', 'Camembert', 'Cheddar', 'Cottage', 'Feta', 'Gouda', 'Mozzarella',
#                        'Parmesan', 'Pasta filata', 'Swiss Cheese', 'Tomme', 'Pecorino']
#         predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]
#         predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]

#         return jsonify({'prediction': predicted_class_names})  # Return the predictions as a JSON response

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500  # Return the error message as a JSON response

def fill_missing_values(encoded_input, df):
    # Convert to NumPy array, handling None
    encoded_input = np.array([x if x is not None else np.nan for x in encoded_input], dtype=float)

    # Check for NaN values
    if np.any(np.isnan(encoded_input)):
        # Fill missing values using mode of the corresponding column
        for i in range(len(encoded_input)):
            if np.isnan(encoded_input[i]):
                # Get mode of the column from the original DataFrame
                mode_values = df.iloc[:, i].mode()  # Get mode of the column
                print(mode_values)
                if not mode_values.empty:  # Ensure there is at least one mode value
                    encoded_input[i] = mode_values[0]  # Use the first mode value
                    print(encoded_input)

    return encoded_input

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
        headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

    try:
        # Handle the actual POST request
        data = request.get_json(force=True)  # Get JSON data from the request
        input_data = data['input']  # Expecting a list of strings or nulls

        # Convert input values using static_mapping and fuzzy matching
        encoded_input = []
        for col_name, value in zip(en.static_mapping.keys(), input_data[0]):
            if value is None:  # If the value is null, keep it as is
                encoded_input.append(None)
            elif value in en.static_mapping[col_name].keys():  # Exact match
                encoded_input.append(en.static_mapping[col_name][value])
            else:  # Use fuzzy matching to find the closest value
                closest_match = get_closest_match(value, en.static_mapping[col_name].keys())
                if closest_match:
                    encoded_input.append(en.static_mapping[col_name][closest_match])
                else:
                    encoded_input.append(None)  # Keep as None if no match is found

        # Convert to NumPy array
        encoded_input = np.array(encoded_input, dtype=object).reshape(1, -1)

        # Fill missing values in the encoded input
        filled_input = fill_missing_values(encoded_input[0], df)

        # Make sure filled_input is a NumPy array before prediction
        filled_input = np.array(filled_input, dtype=float).reshape(1, -1)

        # Predict using the loaded random forest classifier
        predictions = loaded_rf_classifier.predict(filled_input)

        # Convert predictions to class names
        class_names = ['Blue', 'Brie', 'Camembert', 'Cheddar', 'Cottage', 'Feta', 'Gouda', 'Mozzarella',
                       'Parmesan', 'Pasta filata', 'Swiss Cheese', 'Tomme', 'Pecorino']
        predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]
        predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]

        return jsonify({'prediction': predicted_class_names})  # Return the predictions as a JSON response

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return the error message as a JSON response


@app.route('/')
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # เปิดให้เข้าถึงจากทุกอินเทอร์เฟซที่พอร์ต 5000
