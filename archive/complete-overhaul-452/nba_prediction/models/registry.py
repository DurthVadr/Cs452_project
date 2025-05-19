"""
Model registry for storing and retrieving trained models.
"""
import os
import json
import joblib
from datetime import datetime
import pandas as pd

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists

logger = get_logger(__name__)

class ModelRegistry:
    """
    Registry for managing trained models with versioning and metadata.
    """
    def __init__(self, registry_path=None):
        """Initialize the model registry"""
        self.models_dir = config.MODELS_DIR
        ensure_directory_exists(self.models_dir)
        
        self.registry_path = registry_path or os.path.join(self.models_dir, "registry.json")
        self._registry = self._load_registry()
        
    def _load_registry(self):
        """Load the registry from disk or initialize a new one"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": {}}
            
    def _save_registry(self):
        """Save the registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)
            
    def register_model(self, model_name, model, metadata=None):
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model: The trained model object to save
            metadata: Dictionary with additional metadata
            
        Returns:
            dict with model registration info
        """
        metadata = metadata or {}
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version from timestamp
        version = timestamp
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join(self.models_dir, model_name)
        ensure_directory_exists(model_dir)
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}_{version}.pkl")
        joblib.dump(model, model_path)
        
        # Save any additional artifacts
        artifacts = {}
        for name, artifact in metadata.pop("artifacts", {}).items():
            artifact_path = os.path.join(model_dir, f"{model_name}_{version}_{name}.pkl")
            joblib.dump(artifact, artifact_path)
            artifacts[name] = artifact_path
            
        # Create entry in registry
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = {"versions": []}
            
        model_info = {
            "version": version,
            "path": model_path,
            "created_at": timestamp,
            "artifacts": artifacts
        }
        
        # Add metadata
        model_info.update(metadata)
        
        # Add to registry
        self._registry["models"][model_name]["versions"].append(model_info)
        self._registry["models"][model_name]["latest"] = version
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {model_name} (version {version})")
        return model_info
        
    def load_model(self, model_name, version="latest"):
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model to load
            version: Version of the model (default: latest)
            
        Returns:
            The loaded model object
        """
        if model_name not in self._registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version == "latest":
            version = self._registry["models"][model_name]["latest"]
        
        # Find the model version
        model_info = None
        for v in self._registry["models"][model_name]["versions"]:
            if v["version"] == version:
                model_info = v
                break
                
        if model_info is None:
            raise ValueError(f"Version {version} of model {model_name} not found")
            
        # Load the model
        model_path = model_info["path"]
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model {model_name} (version {version})")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name} (version {version}): {e}")
            raise
    
    def load_artifact(self, model_name, artifact_name, version="latest"):
        """
        Load an artifact associated with a model.
        
        Args:
            model_name: Name of the model
            artifact_name: Name of the artifact
            version: Version of the model (default: latest)
            
        Returns:
            The loaded artifact
        """
        if model_name not in self._registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version == "latest":
            version = self._registry["models"][model_name]["latest"]
        
        # Find the model version
        model_info = None
        for v in self._registry["models"][model_name]["versions"]:
            if v["version"] == version:
                model_info = v
                break
                
        if model_info is None:
            raise ValueError(f"Version {version} of model {model_name} not found")
            
        # Check if artifact exists
        if "artifacts" not in model_info or artifact_name not in model_info["artifacts"]:
            raise ValueError(f"Artifact {artifact_name} not found for model {model_name} (version {version})")
            
        # Load the artifact
        artifact_path = model_info["artifacts"][artifact_name]
        try:
            artifact = joblib.load(artifact_path)
            logger.info(f"Loaded artifact {artifact_name} for model {model_name} (version {version})")
            return artifact
        except Exception as e:
            logger.error(f"Error loading artifact {artifact_name} for model {model_name} (version {version}): {e}")
            raise
            
    def list_models(self):
        """List all models in the registry"""
        return list(self._registry["models"].keys())
        
    def get_model_info(self, model_name, version="latest"):
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: latest)
            
        Returns:
            Dictionary with model info
        """
        if model_name not in self._registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version == "latest":
            version = self._registry["models"][model_name]["latest"]
        
        # Find the model version
        for v in self._registry["models"][model_name]["versions"]:
            if v["version"] == version:
                return v
                
        raise ValueError(f"Version {version} of model {model_name} not found")