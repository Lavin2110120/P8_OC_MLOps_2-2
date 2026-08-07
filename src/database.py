import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

DEFAULT_DB_URL = "postgresql+asyncpg://p8_db_ba6y_user:xPkdpcurgzpMopfxc8JRAnvEspmurHib@dpg-d9q6o5lbedkc73b78mtg-a.frankfurt-postgres.render.com/p8_db_ba6y?ssl=require"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Engine asynchrone avec NullPool
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