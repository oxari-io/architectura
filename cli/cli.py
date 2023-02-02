# import fire
import typer
import cloudpickle as pickle
import io
import pandas
from base import OxariMetaModel
from enum import Enum
from typing import Optional
from pyfiglet import Figlet
import pathlib
from loguru import logger
import jsonlines

app = typer.Typer(name="oxari-cli")
f = Figlet(font='slant')
print(f.renderText('OXARI B.V.'))
print("This software is distributed under the Oxari B.V. Software License and therefore forbidden from redistribution.")


class Pipeline(str, Enum):
    one = "1"
    two = "2"
    three = "3"
    all = "all"


class FileType(str, Enum):
    csv = "csv"
    json = "json"

#  TODO: Add progress bar -> https://typer.tiangolo.com/tutorial/progressbar/
@app.command(name="predict")
@logger.catch
def predict(data_path: pathlib.Path = typer.Argument(..., help="Path to csv with features used to predict the scopes."),
            model_path: pathlib.Path = typer.Argument(None, help="Path for the model. No argument chooses the model used in the distribution."),
            confidence: bool = typer.Argument(False, help="Output upper and lower confidence bounds to the prediction."),
            pipeline: Optional[Pipeline] = typer.Argument(Pipeline.all, help="Selects a specific scope to predict."),
            output: str = typer.Argument(None, help="File to write the results into. Filetype is infered from file extension. (csv, json and jsonl is supported)")):
    logger.info(f'Reading data... {data_path}')
    data = pandas.read_csv(data_path)
    logger.info(f'Reading data... SUCCESS - {len(data)} datapoints')
    logger.info(f'Loading model... {model_path or "default"}')
    model_path = model_path or 'local/objects/meta_model/meta_model_test_28-01-2023.pkl'
    logger.info('Loading model... SUCCESS')
    model: OxariMetaModel = pickle.load(io.open(model_path, mode='rb'))
    logger.info('Run prediction...')
    results = model.predict(data, scope=pipeline, return_ci=confidence)
    if not output:
        for index, row in results.iterrows():
            print(row.to_frame().T)
    if output and output.endswith(".csv"):
        results.to_csv(output)
        logger.info(f"Results are written to => {pathlib.Path(output).absolute()}")
    if output and output.endswith(".jsonl"):
        writer = jsonlines.Writer(io.open(output, mode="w"))
        results_json_l = results.to_dict(orient='records')
        writer.write_all(results_json_l)
        logger.info(f"Results are written to => {pathlib.Path(output).absolute()}")
    if output and output.endswith(".json"):
        results.to_json(output, orient='records')
        logger.info(f"Results are written to => {pathlib.Path(output).absolute()}")
    logger.info('DONE')


@app.command(name="license")
def license():
    license_path = pathlib.Path('license.txt')
    current_path = pathlib.Path(__file__)
    f = io.open(current_path.parent / license_path)
    print(f.read())
    f.close()


if __name__ == '__main__':
    app()
