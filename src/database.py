import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

# URL de connexion PostgreSQL (Driver asyncpg)
DATABASE_URL = os.getenv("DATABASE_URL")

# Engine asynchrone avec NullPool (idéal pour la compatibilité asyncio/pytest et conteneurs)
engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool,
    echo=False
)

# Fabrique de sessions asynchrones
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Classe de base pour les modèles ORM
class Base(DeclarativeBase):
    pass

# Dependency Injection pour FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session