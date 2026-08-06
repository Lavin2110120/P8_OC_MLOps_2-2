import os
<<<<<<< HEAD
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

DEFAULT_DB_URL = "postgresql+asyncpg://p8_oc_mlops_part2_db_user:llJ81uZProdA5WKLyn6xs3PEldHsa0mo@dpg-daaive942hec73akonog-a.frankfurt-postgres.render.com/p8_oc_mlops_part2_db?ssl=require"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Engine asynchrone avec NullPool
engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool,
    echo=False
=======
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# URL de connexion PostgreSQL (Driver asyncpg)
# Format : postgresql+asyncpg://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:postgres@localhost:5432/scoring_db"
)

# Moteur de connexion asynchrone avec pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Mettre à True en dev si tu veux voir les requêtes SQL générées
    pool_size=10,
    max_overflow=20
>>>>>>> 95a751e (adding performance tests and reports)
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

