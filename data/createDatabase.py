import sqlalchemy as sa
import sqlite3
from data_model import tables
from sqlalchemy.orm import Session
from datetime import datetime

db_connection = 'sqlite:///priceData.db'

engine = sa.create_engine(db_connection, future=True)

tables.mapper_registry.metadata.drop_all(engine)

tables.mapper_registry.metadata.create_all(engine)