import joblib

with open("local/objects/MetaModel_22-11-2022.pkl", "rb") as handle:
    m = joblib.load(handle)


print(m.__dict__)