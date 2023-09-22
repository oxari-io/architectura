# architectura

Contains the skeletons for the pipeline architecture

# Environment variables

These variables need to be set in order to get the data saving process to work.

| Variable                  | Example                                            | Description                                  |
| ------------------------- | -------------------------------------------------- | -------------------------------------------- |
| `S3_KEY_ID`               | `AKIAIOSFODNN7EXAMPLE`                             | The AWS access key ID for S3 access.         |
| `S3_ACCESS_KEY`           | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`         | The AWS secret access key for S3 access.     |
| `S3_ENDPOINT`             | `https://ams3.digitaloceanspaces.com`              | The endpoint URL of the S3 service.          |
| `S3_REGION`               | `ams3`                                             | The AWS region where your S3 bucket resides. |
| `S3_BUCKET`               | `oxari-storage`                                    | The name of the S3 bucket.                   |
| `MONGO_CONNECTION_STRING` | `mongodb://username:password@localhost:27017/mydb` | Connection string to MongoDB server.         |
| `MONGO_DATABASE_NAME`        | `d_data`                                              | The MongoDB database to use.               |


# DataLoader

Be aware of the misleading name, it's more like a DataManager


# OxariMixin

## REVIEW-ME!

I removed Mixin from where it was not needed
I removed run abstract method from OxariMixin

# Install

## PyGraphviz

This is more difficult to install on windows.

```bash
poetry shell
pip install --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz
```

This is for macOS installation.

```bash
poetry shell
pip install --use-pep517 \
            --config-setting="--global-option=build_ext" \
            --config-setting="--build-option=-I$(brew --prefix graphviz)/include/" \
            --config-setting="--build-option=-L$(brew --prefix graphviz)/lib/" \
            pygraphviz
```
