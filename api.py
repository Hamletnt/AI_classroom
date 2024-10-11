from flask import Flask, request, jsonify
import boto3
import joblib
import numpy as np
import os

app = Flask(__name__)

# ประกาศคลาสที่จำเป็น
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifier:
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
        best_split = {}
        max_info_gain = -float("inf")
    
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
        
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
            
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain

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
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

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

# การตั้งค่า AWS S3
s3 = boto3.client('s3')
bucket_name = 'your-bucket-name'  # เปลี่ยนเป็นชื่อบัคเก็ตของคุณ
model_file = 'path/to/your/model.joblib'  # เปลี่ยนเป็นพาธของไฟล์โมเดลใน S3
local_model_path = '/tmp/model.joblib'  # พาธสำหรับเก็บโมเดลที่ดาวน์โหลดลงเครื่อง

# ฟังก์ชันโหลดโมเดลจาก S3
def load_model_from_s3():
    if not os.path.exists(local_model_path):
        s3.download_file(bucket_name, model_file, local_model_path)
    return joblib.load(local_model_path)

# โหลดโมเดลที่ถูกฝึกแล้ว
model = load_model_from_s3()

# สร้าง route สำหรับการทำนาย
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # รับข้อมูล JSON ที่ส่งมา
    features = np.array(data['features'])  # สมมติว่า 'features' คือ key ที่เก็บข้อมูลฟีเจอร์
    predictions = model.predict(features)  # ทำนายด้วยโมเดล
    return jsonify({'predictions': predictions.tolist()})  # ส่งผลลัพธ์กลับไป

if __name__ == '__main__':
    app.run(debug=True)