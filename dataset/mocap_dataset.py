import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from dotenv import load_dotenv
from scipy import stats


class MocapDataset(Dataset):
    def __init__(self, uri , db_name, collection_name, skeleton):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._skeleton = skeleton

        self.client = None
        self.collection = None

        self.joint_names = []
        self.num_joints = 0
        self._ids = []
        self.total = 0

        self._initialize_ids_and_metadata()

    def _initialize_ids_and_metadata(self):
        load_dotenv()
        self._connect()
        sample_doc = self.collection.find_one()
        if not sample_doc:
            raise ValueError("The collection is empty.")
        if 'kps_2d' not in sample_doc:
            raise ValueError("Documents must contain 'kps_2d' field.")

        all_joints = list(sample_doc['kps_2d'].keys())
        self.joint_names = [joint for joint in all_joints if joint.strip().lower() not in ['date', 'body']]
        self.num_joints = len(self.joint_names)

        self._ids = list(self.collection.find({}, {'_id': 1}))
        self._ids = [doc['_id'] for doc in self._ids]
        self.total = len(self._ids)

    def _connect(self):
        if self.client is None:
            try:
                # Use the URI directly instead of specifying port separately
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                # Test the connection
                self.client.server_info()
                db = self.client[self.db_name]
                self.collection = db[self.collection_name]
            except Exception as e:
                raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}\nPlease check:\n1. MongoDB server is running\n2. MONGO_URI in .env is correct\n3. Database and collection names are correct")

    def parse_keypoints(self, keypoints_dict):
        keypoints_flat = []
        for joint in self.joint_names:
            coords = keypoints_dict.get(joint, [0.0, 0.0])
            if len(coords) < 2:
                coords = [0.0, 0.0]
            keypoints_flat.extend(coords[:2])
        
        keypoints_array = np.array(keypoints_flat, dtype=np.float32)
        return self.set_precision(data=keypoints_array)

    def parse_keypoints_3d(self, keypoints_dict):
        keypoints_flat = []
        for joint in self.joint_names:
            coords = keypoints_dict.get(joint, [0.0, 0.0, 0.0])
            if len(coords) < 3:
                coords = [0.0, 0.0, 0.0]
            keypoints_flat.extend(coords[:3])
        
        keypoints_array = np.array(keypoints_flat, dtype=np.float32)
        return self.set_precision(data=keypoints_array)

    def set_precision(self, data=None, precision=4):
        if data is not None:
            return np.round(data, decimals=precision)
        
        if hasattr(self, 'camera_matrix'):
            self.cameras = np.round(self.cameras, decimals=precision)
        if hasattr(self, 'keypoints'):
            self.keypoints = np.round(self.keypoints, decimals=precision)

    def clip_outliers_camera_matrix(self, camera_matrix):
        # Ensure camera_matrix is the right shape (4,4)
        camera_matrix = camera_matrix.reshape(4, 4)
        
        # Create a copy to avoid modifying the original
        cleaned_camera_matrix = camera_matrix.copy()
        
        # Get the translation vector (last row)
        translation = camera_matrix[3, :]
        
        # Define clip bounds (e.g., 5th and 95th percentiles)
        lower_bound = np.percentile(translation, 5)
        upper_bound = np.percentile(translation, 95)
        
        # Clip the translation values
        clipped_translation = np.clip(translation, lower_bound, upper_bound)
        
        # Replace the last row with clipped values
        cleaned_camera_matrix[3, :] = clipped_translation
        
        # Flatten the cleaned camera matrix
        cleaned_camera_matrix = cleaned_camera_matrix.flatten()
        
        return cleaned_camera_matrix
              

    def parse_camera_matrix(self, camera_matrix_list):
        if len(camera_matrix_list) != 16:
            raise ValueError("Camera matrix must have 16 elements.")
        
        camera_matrix = np.array(camera_matrix_list, dtype=np.float32)
        camera_matrix = self.set_precision(data=camera_matrix)
        
        return self.clip_outliers_camera_matrix(camera_matrix)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.total}.")

        self._connect()  
        _id = self._ids[idx]
        document = self.collection.find_one({'_id': _id})

        if not document:
            raise ValueError(f"No document found with _id: {_id}")

        # Get 2D keypoints
        keypoints_dict = document.get('kps_2d', {})
        keypoints_flattened = self.parse_keypoints(keypoints_dict)

        # Get 3D keypoints
        keypoints_dict_3d = document.get('kps_3d', {})
        keypoints_flattened_3d = self.parse_keypoints_3d(keypoints_dict_3d)

        camera_matrix_list = document.get('camera_matrix', [])
        camera_matrix = self.parse_camera_matrix(camera_matrix_list)

        # Validate 2D keypoints
        expected_length = self.num_joints * 2
        actual_length = keypoints_flattened.shape[0]
        if actual_length != expected_length:
            missing_joints = [joint for joint in self.joint_names if joint not in keypoints_dict]
            if missing_joints:
                print(f"Warning: Missing joint coordinates for: {missing_joints}")
            raise ValueError(f"Sample index {idx} has {actual_length} 2D keypoints, expected {expected_length}.")

        # Validate 3D keypoints
        expected_length_3d = self.num_joints * 3
        actual_length_3d = keypoints_flattened_3d.shape[0]
        if actual_length_3d != expected_length_3d:
            missing_joints = [joint for joint in self.joint_names if joint not in keypoints_dict_3d]
            if missing_joints:
                print(f"Warning: Missing 3D joint coordinates for: {missing_joints}")
            raise ValueError(f"Sample index {idx} has {actual_length_3d} 3D keypoints, expected {expected_length_3d}.")

        keypoints_tensor = torch.from_numpy(keypoints_flattened).float()
        keypoints_tensor_3d = torch.from_numpy(keypoints_flattened_3d).float()
        camera_matrix_tensor = torch.from_numpy(np.array(camera_matrix, dtype=np.float32)).float()
        
        return keypoints_tensor, keypoints_tensor_3d, camera_matrix_tensor, self.joint_names, list(range(len(self.joint_names)))

    def close_connection(self):
        if self.client:
            self.client.close()
            self.client = None
