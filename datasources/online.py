from os import environ as env

import boto3
import botocore
import pandas as pd
from typing_extensions import Self
import requests
from base.dataset_loader import Datasource
from requests.models import Response


# TODO: The internet loaders need a progressbar
class S3Datasource(Datasource):

    def __init__(self, **kwargs):
        # path needs to be the location of the file UNDER the bucket
        super().__init__(**kwargs)
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')  # Endpoint ${REGION}.digitaloceanspaces.com
        self.do_spaces_bucket = env.get('S3_BUCKET')  # DO-Space
        self.do_spaces_region = env.get('S3_REGION')  # Repetition of ${REGION}
        self.connect()

    def _check_if_data_exists(self) -> bool:
        self.meta = self.client.head_object(Bucket=self.do_spaces_bucket, Key=self.path)

    def _load(self) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        response = self.client.get_object(Bucket=self.do_spaces_bucket, Key=self.path)
        self._data = pd.read_csv(response['Body'])
        return self

    def connect(self):
        self.session: boto3.Session = boto3.Session()
        self.client = self.session.client(
            's3',
            config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
            region_name=self.do_spaces_region,
            endpoint_url=self.do_spaces_endpoint,
            aws_access_key_id=self.do_spaces_key_id,
            aws_secret_access_key=self.do_spaces_access_key,
        )
        return self.client


class OnlineDatasource(Datasource):

    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(path, **kwargs)

    def _check_if_data_exists(self) -> bool:
        self.meta = requests.head(self.path).headers
        return True

    def _load(self) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        response = requests.get(self.path, allow_redirects=True)
        self._data = self._handle_file(response)
        return self

    def _handle_file(self, response: Response):
        return response.content


class OnlineCSVDatasource(OnlineDatasource):

    def _handle_file(self, response: Response):
        return pd.read_csv(response.content)


class OnlineExcelDatasource(OnlineDatasource):

    def _handle_file(self, response: Response):
        return pd.read_excel(response.content)