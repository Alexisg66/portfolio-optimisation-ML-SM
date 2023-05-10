import sqlalchemy as sa
from sqlalchemy import orm
import datetime as dt

mapper_registry = orm.registry()

@mapper_registry.mapped
class Load:
    __tablename__ = 'Load'
    loadId: int = sa.Column(sa.Integer, primary_key=True)
    loadAt: dt.datetime = sa.Column(sa.DateTime, default=sa.func.now())
    loadType: str = sa.Column(sa.NVARCHAR(20))

@mapper_registry.mapped
class Ticker:
    __tablename__ = 'Ticker'
    ticker = sa.Column(sa.NVARCHAR(4),primary_key = True)
    name = sa.Column(sa.NVARCHAR(100))
    loadId = sa.Column(sa.Integer, sa.ForeignKey('Load.loadId'))
    
    load = sa.orm.relationship("Load", backref='ticker')

@mapper_registry.mapped
class PriceData:
    __tablename__ = 'PriceData'
    loadId = sa.Column(sa.Integer, sa.ForeignKey('Load.loadId'))
    ticker = sa.Column(sa.NVARCHAR(4), sa.ForeignKey('Ticker.ticker'), primary_key=True)
    date: dt.datetime = sa.Column(sa.DateTime, primary_key = True)
    close = sa.Column(sa.Numeric(9,2,asdecimal=False))
    volume = sa.Column(sa.Integer)
    high = sa.Column(sa.Numeric(9,2,asdecimal=False))
    low = sa.Column(sa.Numeric(9,2,asdecimal=False))


    load = sa.orm.relationship("Load", backref='data')


    