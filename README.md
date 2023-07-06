# architectura
Contains the skeletons for the pipeline architecture


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