from enum import Enum


class DTypes(Enum):
    CATEGORY = "category"
    NUMERICAL = "numerical"
    META = "meta"

class MTypes(Enum):
    KEY = "key"
    FEATURE = "feat"
    TARGET = "tgt"

class ColType:
    def __init__(self, name, dtype, mtype):
        self.name = name
        self.dtype = dtype
        self.mtype = mtype

class ColumnRegistry(Enum):
    isin = ColType('isin', DTypes.META, MTypes.KEY)
    year = ColType('year', DTypes.META, MTypes.KEY)
    scope_1 = ColType('scope_1', DTypes.NUMERICAL, MTypes.TARGET)
    scope_2 = ColType('scope_2', DTypes.NUMERICAL, MTypes.TARGET)
    scope_3 = ColType('scope_3', DTypes.NUMERICAL, MTypes.TARGET)
    revenue = ColType('revenue', DTypes.NUMERICAL, MTypes.FEATURE)
    equity = ColType('equity', DTypes.NUMERICAL, MTypes.FEATURE)
    employees = ColType('employees', DTypes.NUMERICAL, MTypes.FEATURE)
    ppe = ColType('ppe', DTypes.NUMERICAL, MTypes.FEATURE)
    inventories = ColType('inventories', DTypes.NUMERICAL, MTypes.FEATURE)
    roa = ColType('roa', DTypes.NUMERICAL, MTypes.FEATURE)
    roe = ColType('roe', DTypes.NUMERICAL, MTypes.FEATURE)
    cash = ColType('cash', DTypes.NUMERICAL, MTypes.FEATURE)
    rd_expenses = ColType('rd', DTypes.NUMERICAL, MTypes.FEATURE)
    stock_return = ColType('stock_return', DTypes.NUMERICAL, MTypes.FEATURE)
    market_cap = ColType('market_cap', DTypes.NUMERICAL, MTypes.FEATURE)
    net_income = ColType('net_income', DTypes.NUMERICAL, MTypes.FEATURE)
    total_assets = ColType('total_assets', DTypes.NUMERICAL, MTypes.FEATURE)
    total_liab = ColType('total_liab', DTypes.NUMERICAL, MTypes.FEATURE)
    industry_name = ColType('industry', DTypes.CATEGORY, MTypes.FEATURE)
    sector_name = ColType('sector', DTypes.CATEGORY, MTypes.FEATURE)
    country_name = ColType('country', DTypes.CATEGORY, MTypes.FEATURE)

    @staticmethod
    def get_features():
        return [col.name for col in ColumnRegistry if col.value.mtype == MTypes.FEATURE]    
    
    @staticmethod
    def get_targets():
        return [col.name for col in ColumnRegistry if col.value.mtype == MTypes.TARGET]  
    
 
    
      

class NumMapping(Enum):
    scope_1 = "scope_1"
    scope_2 = "scope_2"
    scope_3 = "scope_3"
    revenue = "revenue"
    equity = "equity"
    stock_return = "stock_return"
    market_cap = "market_cap"
    employees = "employees"
    ppe = "ppe"
    inventories = "inventories"
    roa = "roa"
    roe = "roe"
    rd = "rd"
    net_income = "net_income"
    total_assets = "total_assets"
    cash = "cash"
    total_liab = "total_liab"



    @staticmethod
    def get_features():
        return [cat.name for cat in NumMapping if "scope_" not in cat.name]

    @staticmethod
    def get_targets():
        return [cat.name for cat in NumMapping if "scope_" in cat.name]

class CatMapping(Enum):
    isin = "isin"
    industry_name = "industry"
    sector_name = "sector"
    country_name = "country"

    @staticmethod
    def to_dict(reverse=False):
        return {cat.value: cat.name for cat in CatMapping} if reverse else {cat.name: cat.value for cat in Categories1}

    @staticmethod
    def get_features():
        return [cat.name for cat in CatMapping if cat.name not in ["isin"]]
    

