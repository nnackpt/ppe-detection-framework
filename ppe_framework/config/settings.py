"""
PPE Detection Framework - Configuration Settings 
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os

@dataclass
class ModelConfig:
    person_model_path: str
    ppe_model_path: str
    classification_model_path: Optional[str] = None
    device: str = "cuda"
    confidence_threshold: float = 0.7
    classification_threshold: float = 0.6
    use_half_precision: bool = True
    
@dataclass
class CameraConfig:
    camera_urls: List[str] = field(default_factory=list)
    roi_zones: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    exclusion_zones: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    detection_interval: int = 10
    camera_off: int = 5
    
@dataclass
class NotificationConfig:
    enable_email: bool = True
    enable_sound: bool = True
    email_cooldown: int = 300
    sound_cooldown: int = 10
    smtp_server: str = "smtp.alv.autoliv.int"
    smtp_port: int = 25
    sender_email: str = ""
    sender_password: str = ""
    recipient_emails: List[str] = field(default_factory=list)
    
@dataclass
class DatabaseConfig:
    server: str = ""
    database: str = ""
    username: str = ""
    password: str = ""
    driver: str = "ODBC Driver 18 for SQL Server"
    
@dataclass
class PPEConfig:
    model: ModelConfig
    camera: CameraConfig
    notification: NotificationConfig
    database: Optional[DatabaseConfig] = None
    
    ng_save_dir: str = "ng_images"
    ng_cooldown: int = 5
    save_original: bool = True
    save_annotated: bool = True
    consecutive_ng_threshold: int = 3
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        model_config = ModelConfig(**config_dict.get("model", {}))
        camera_config = CameraConfig(**config_dict.get("camera", {}))
        notification_config = NotificationConfig(**config_dict.get("notification", {}))
        
        db_config = None
        if "database" in config_dict:
            db_config = DatabaseConfig(**config_dict["database"])
            
        return cls(
            model=model_config,
            camera=camera_config,
            notification=notification_config,
            database=db_config,
            **{k: v for k, v in config_dict.items()
               if k not in ["model", "camera", "notification", "database"]}
        )
        
    @classmethod
    def from_env(cls):
        model_config = ModelConfig(
            person_model_path=os.getenv("PERSON_MODEL_PATH", "models/person.pt"),
            ppe_model_path=os.getenv("PPE_MODEL_PATH", "models/ppe.pt"),
            classification_model_path=os.getenv("CLASSIFICATION_MODEL_PATH"),
            device=os.getenv("DEVICE", "cuda"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            classification_threshold=float(os.getenv("CLASSIFICATION_THRESHOLD", "0.6"))
        )
        
        camera_config = CameraConfig()
        notification_config = NotificationConfig(
            sender_email=os.getenv("SENDER_EMAIL", ""),
            sender_password=os.getenv("SENDER_PASSWORD", "")
        )
        
        db_config = None
        if os.getenv("DB_SERVER"):
            db_config = DatabaseConfig(
                server=os.getenv("DB_SERVER"),
                database=os.getenv("DB_DATABASE"),
                username=os.getenv("DB_USERNAME"),
                password=os.getenv("DB_PASSWORD")
            )
            
        return cls(
            model=model_config,
            camera=camera_config,
            notification=notification_config,
            database=db_config
        )