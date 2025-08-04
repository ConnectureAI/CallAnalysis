"""
Database connection and session management for the call analysis system.

This module provides async database connectivity using SQLAlchemy with connection
pooling, health checks, and proper session management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, pool
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from .config import get_settings
from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def create_engine() -> AsyncEngine:
    """Create and configure the database engine."""
    settings = get_settings()
    
    # Connection pool configuration
    if settings.is_development:
        # Use NullPool for development to avoid connection issues
        poolclass = NullPool
        pool_kwargs = {}
    else:
        # Use QueuePool for production
        poolclass = QueuePool
        pool_kwargs = {
            "pool_size": settings.database.pool_size,
            "max_overflow": settings.database.max_overflow,
            "pool_timeout": settings.database.pool_timeout,
            "pool_recycle": settings.database.pool_recycle,
            "pool_pre_ping": True,  # Verify connections before use
        }
    
    # Engine configuration
    engine_kwargs = {
        "url": settings.database.url,
        "echo": settings.debug,
        "echo_pool": settings.debug,
        "poolclass": poolclass,
        "future": True,
        **pool_kwargs
    }
    
    # SSL configuration
    if settings.database.ssl_cert:
        connect_args = {
            "server_settings": {
                "application_name": settings.app_name,
            }
        }
        if settings.database.ssl_mode != "disable":
            connect_args["ssl"] = {
                "sslmode": settings.database.ssl_mode,
            }
            if settings.database.ssl_cert:
                connect_args["ssl"]["sslcert"] = settings.database.ssl_cert
            if settings.database.ssl_key:
                connect_args["ssl"]["sslkey"] = settings.database.ssl_key
            if settings.database.ssl_ca:
                connect_args["ssl"]["sslrootcert"] = settings.database.ssl_ca
        
        engine_kwargs["connect_args"] = connect_args
    
    engine = create_async_engine(**engine_kwargs)
    
    # Add connection event listeners
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set connection-level settings."""
        if "postgresql" in str(engine.url):
            # Set PostgreSQL-specific settings
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET timezone='UTC'")
                cursor.execute("SET statement_timeout='300s'")  # 5 minutes
    
    @event.listens_for(engine.sync_engine, "checkout")
    def checkout_listener(dbapi_connection, connection_record, connection_proxy):
        """Log connection checkout in debug mode."""
        if settings.debug:
            logger.debug(f"Connection checked out: {id(dbapi_connection)}")
    
    @event.listens_for(engine.sync_engine, "checkin")
    def checkin_listener(dbapi_connection, connection_record):
        """Log connection checkin in debug mode."""
        if settings.debug:
            logger.debug(f"Connection checked in: {id(dbapi_connection)}")
    
    return engine


def get_engine() -> AsyncEngine:
    """Get the global database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get the global session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
    return _async_session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with proper error handling and cleanup.
    
    Usage:
        async with get_db_session() as session:
            # Use session here
            result = await session.execute(query)
            await session.commit()
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def get_db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage in FastAPI endpoints:
        async def endpoint(session: AsyncSession = Depends(get_db_session_dependency)):
            # Use session here
    """
    async with get_db_session() as session:
        yield session


class DatabaseManager:
    """Database manager for handling connections and operations."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker[AsyncSession]] = None
    
    async def initialize(self) -> None:
        """Initialize the database connection."""
        logger.info("Initializing database connection...")
        self._engine = create_engine()
        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        # Test the connection
        try:
            async with self._session_maker() as session:
                await session.execute("SELECT 1")
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            logger.info("Closing database connection...")
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if not self._session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self._session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Perform a database health check."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the database is initialized."""
        return self._engine is not None and self._session_maker is not None


# Global database manager instance
db_manager = DatabaseManager()


async def create_tables() -> None:
    """Create all database tables."""
    engine = get_engine()
    logger.info("Creating database tables...")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")


async def drop_tables() -> None:
    """Drop all database tables."""
    engine = get_engine()
    logger.info("Dropping database tables...")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("Database tables dropped successfully")


async def init_db() -> None:
    """Initialize the database with tables and default data."""
    await create_tables()
    logger.info("Database initialized successfully")


async def cleanup_db() -> None:
    """Cleanup database connections."""
    global _engine, _async_session_maker
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
    
    await db_manager.close()
    logger.info("Database cleanup completed")


# Health check functions
async def check_database_health() -> dict:
    """Check database health and return status information."""
    try:
        async with get_db_session() as session:
            # Test basic connectivity
            result = await session.execute("SELECT 1 as test")
            test_value = result.scalar()
            
            # Get database version
            version_result = await session.execute("SELECT version()")
            db_version = version_result.scalar()
            
            # Get connection count (PostgreSQL specific)
            try:
                conn_result = await session.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                active_connections = conn_result.scalar()
            except Exception:
                active_connections = "unknown"
            
            return {
                "status": "healthy",
                "test_query": test_value == 1,
                "database_version": db_version,
                "active_connections": active_connections,
                "engine_pool_size": _engine.pool.size() if _engine else 0,
                "engine_checked_out": _engine.pool.checkedout() if _engine else 0,
            }
    
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "test_query": False,
        }


# Context managers for application lifecycle
@asynccontextmanager
async def database_lifespan():
    """Context manager for database lifecycle management."""
    try:
        # Initialize database
        await db_manager.initialize()
        yield db_manager
    finally:
        # Cleanup database connections
        await cleanup_db()


# Dependency injection helpers
async def get_database_manager() -> DatabaseManager:
    """Get the database manager instance."""
    if not db_manager.is_initialized:
        await db_manager.initialize()
    return db_manager


# Migration helpers (to be used with Alembic)
def get_database_url_for_alembic() -> str:
    """Get database URL for Alembic migrations."""
    settings = get_settings()
    return settings.database.sync_url  # Alembic needs synchronous URL