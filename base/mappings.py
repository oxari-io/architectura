from enum import Enum

CATEGORIES_STRING = "categories"

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
    rd_expenses = "rd_expenses"
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
    

