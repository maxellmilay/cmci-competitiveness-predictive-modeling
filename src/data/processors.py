"""
Example Data Processor Implementations
Implement the DataProcessor protocol for your specific data needs.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging


class BaseDataProcessor:
    """Base class with common data processing utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values with specified strategy"""
        if strategy == "drop":
            return data.dropna()
        elif strategy == "fillna":
            # Use different strategies for numeric and categorical
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            
            if len(numeric_cols) > 0:
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            if len(categorical_cols) > 0:
                data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        elif strategy == "impute":
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='mean')
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        feature_cols = [col for col in data.columns if col != target_col]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        data[feature_cols] = self.scaler.fit_transform(data[feature_cols])
        return data


class CMCIDataProcessor(BaseDataProcessor):
    """Data processor for CMCI competitiveness data"""
    
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load CMCI data"""
        data_path = config['raw_data_path']
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            self.logger.info(f"Loaded data shape: {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def preprocess(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess CMCI data"""
        self.logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original
        processed_data = data.copy()
        
        # Handle missing values
        preprocessing_config = config.get('preprocessing', {})
        missing_strategy = preprocessing_config.get('missing_value_strategy', 'fillna')
        processed_data = self._handle_missing_values(processed_data, missing_strategy)
        
        # Feature engineering (example)
        if 'competitiveness_score' in processed_data.columns:
            # Create binary target
            processed_data['target'] = (processed_data['competitiveness_score'] > processed_data['competitiveness_score'].median()).astype(int)
        
        # Normalize features if requested
        if preprocessing_config.get('normalize', False):
            processed_data = self._normalize_features(processed_data)
        
        self.logger.info(f"Preprocessed data shape: {processed_data.shape}")
        return processed_data
    
    def split_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        splitting_config = config.get('splitting', {})
        test_size = splitting_config.get('test_size', 0.2)
        random_state = splitting_config.get('random_state', 42)
        
        target_col = 'target'
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        stratify = data[target_col] if splitting_config.get('stratify', False) else None
        
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        self.logger.info(f"Split data - Train: {train_data.shape}, Test: {test_data.shape}")
        return train_data, test_data


class TabularDataProcessor(BaseDataProcessor):
    """Generic processor for tabular data"""
    
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load tabular data from various formats"""
        data_path = config['raw_data_path']
        
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def preprocess(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generic preprocessing for tabular data"""
        processed_data = data.copy()
        
        # Handle missing values
        preprocessing_config = config.get('preprocessing', {})
        missing_strategy = preprocessing_config.get('missing_value_strategy', 'fillna')
        processed_data = self._handle_missing_values(processed_data, missing_strategy)
        
        # Encode categorical variables
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'target':  # Don't encode target if it's categorical
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        
        # Normalize if requested
        if preprocessing_config.get('normalize', False):
            processed_data = self._normalize_features(processed_data)
        
        return processed_data
    
    def split_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data with optional validation set"""
        splitting_config = config.get('splitting', {})
        test_size = splitting_config.get('test_size', 0.2)
        random_state = splitting_config.get('random_state', 42)
        
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        
        return train_data, test_data


class TextDataProcessor(BaseDataProcessor):
    """Data processor for text classification tasks"""
    
    def __init__(self):
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load text data"""
        return pd.read_csv(config['raw_data_path'])
    
    def preprocess(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess text data"""
        processed_data = data.copy()
        
        # Assume text column is named 'text' and target is 'label'
        text_col = config.get('text_column', 'text')
        target_col = config.get('target_column', 'label')
        
        # Vectorize text
        text_features = self.vectorizer.fit_transform(processed_data[text_col])
        
        # Create feature DataFrame
        feature_names = [f"tfidf_{i}" for i in range(text_features.shape[1])]
        feature_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=processed_data.index)
        
        # Combine with target
        result = pd.concat([feature_df, processed_data[target_col].rename('target')], axis=1)
        
        return result
    
    def split_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split text data"""
        splitting_config = config.get('splitting', {})
        test_size = splitting_config.get('test_size', 0.2)
        random_state = splitting_config.get('random_state', 42)
        
        # Stratify by target for classification
        stratify = data['target'] if 'target' in data.columns else None
        
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        return train_data, test_data 
