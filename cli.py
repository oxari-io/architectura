# import fire
import typer
import cloudpickle as pickle
import io
import pandas
from base import OxariMetaModel
from enum import IntEnum, Enum

app = typer.Typer()

class Pipeline(Enum):
    one = "1"
    two = "2"
    three = "3"
    all = "all"

@app.command()
def predict(data_path:str, model_path:str=None, confidence:bool=False, pipeline:Pipeline=Pipeline.all):
    model_path = model_path or 'local/objects/meta_model/meta_model_test_28-01-2023.pkl'
    data = pandas.read_csv(data_path)

    model:OxariMetaModel = pickle.load(io.open(model_path, mode='rb'))
    if pipeline == pipeline.all:
        model.predict(data, return_std=confidence)
    else:
        model.get_pipeline(int(pipeline)).predict(data, return_std=confidence)


if __name__ == '__main__':
  app()