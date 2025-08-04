"""
Configuration management for the call analysis system.

This module handles all configuration settings using Pydantic Settings for
type validation and environment variable loading.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="call_analysis", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    
    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")
    
    # SSL settings
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    ssl_cert: Optional[str] = Field(default=None, description="SSL certificate path")
    ssl_key: Optional[str] = Field(default=None, description="SSL key path")
    ssl_ca: Optional[str] = Field(default=None, description="SSL CA path")
    
    @property
    def url(self) -> str:
        """Get database URL."""
        if self.password:
            return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return f"postgresql+asyncpg://{self.user}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database URL for migrations."""
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    # Connection settings
    max_connections: int = Field(default=50, description="Maximum connections")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout")
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class OpenAISettings(BaseSettings):
    """OpenAI API configuration settings."""
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4-turbo-preview", description="Default model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=1000, ge=1, description="Maximum tokens per request")
    
    # Rate limiting
    requests_per_minute: int = Field(default=500, description="Requests per minute limit")
    tokens_per_minute: int = Field(default=150000, description="Tokens per minute limit")
    
    # Retry settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or not v.startswith("sk-"):
            raise ValueError("Valid OpenAI API key is required")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or console)")
    file: Optional[str] = Field(default=None, description="Log file path")
    max_size: int = Field(default=10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    @validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    model_config = SettingsConfigDict(env_prefix="MONITORING_")
    
    enabled: bool = Field(default=False, description="Enable monitoring")
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    
    # OpenTelemetry settings
    otel_enabled: bool = Field(default=False, description="Enable OpenTelemetry")
    otel_endpoint: Optional[str] = Field(default=None, description="OTEL collector endpoint")
    otel_service_name: str = Field(default="call-analysis", description="Service name for traces")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="Call Analysis System", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Data settings
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    models_dir: Path = Field(default=Path("data/models"), description="ML models directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # Feature flags
    enable_real_time_coaching: bool = Field(default=True, description="Enable real-time coaching")
    enable_predictive_analytics: bool = Field(default=True, description="Enable predictive analytics")
    enable_semantic_analysis: bool = Field(default=True, description="Enable semantic analysis")
    
    # Processing settings
    batch_size: int = Field(default=100, description="Batch processing size")
    max_concurrent_analyses: int = Field(default=10, description="Max concurrent analyses")
    analysis_timeout: int = Field(default=300, description="Analysis timeout in seconds")
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL."""
        return self.database.url if async_driver else self.database.sync_url


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings


# Environment-specific settings
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    environment: str = "development"
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings(level="DEBUG"))


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    environment: str = "production"
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings(level="INFO"))
    
    # Production security requirements
    @validator("openai")
    def validate_production_openai(cls, v):
        if not v.api_key or v.api_key == "":
            raise ValueError("OpenAI API key is required in production")
        return v
    
    @validator("security")
    def validate_production_security(cls, v):
        if not v.secret_key or len(v.secret_key) < 32:
            raise ValueError("Strong secret key is required in production")
        return v


def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings for specific environment."""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "development":
        return DevelopmentSettings()
    else:
        return Settings(environment=env)