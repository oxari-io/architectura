# architectura

A comprehensive machine learning pipeline architecture for carbon emission modeling and portfolio analysis.

> **📋 Technical Report**: `Portfolio__Carbon_Emission_Modelling.pdf` (in the repo root) covers all theoretical aspects of the carbon emission modelling approach, including methodology, algorithms, and mathematical foundations.

## Overview

This project implements a sophisticated machine learning pipeline for predicting carbon emissions across different scopes (Scope 1, 2, and 3) using financial and categorical company data. The architecture features:

- **Multi-scope modeling**: Separate models for different emission categories
- **Advanced feature engineering**: Financial ratio analysis, categorical encoding, and temporal features
- **Ensemble methods**: Mini-model army approach with voting and stacking strategies
- **Confidence estimation**: Built-in uncertainty quantification for predictions
- **Flexible data sources**: Support for both local and cloud-based data storage
- **Production-ready pipeline**: Comprehensive training, evaluation, and inference workflows

## Installation

### Prerequisites
- Python 3.10+
- Poetry (recommended) or Conda + pip
- Optional: Graphviz headers for visualization components

### Quickstart (Poetry)
1. Install dependencies:
   ```bash
   poetry install
   ```
2. Create a `.env` file (see configuration section below).
3. Create required output directories:
   ```bash
   mkdir -p model-data/output local/prod_runs
   ```
4. Train models:
   ```bash
   poetry run python main_prod.py
   ```
5. Run inference:
   ```bash
   poetry run python main_inference_pipeline.py
   ```

### Alternative: Conda + pip
```bash
conda create -n py310 python=3.10 -y
conda activate py310
pip install -r requirements.txt
```

### Server Deployment
For production server deployment on Ubuntu/Debian systems, use the automated setup script:
```bash
bash setup-do-server.sh
```

## Configuration

### Environment Variables
The system requires configuration for data storage and persistence. Create a `.env` file in the project root:

```dotenv
# S3-Compatible Storage (DigitalOcean Spaces, AWS S3, etc.)
S3_KEY_ID=your_access_key
S3_ACCESS_KEY=your_secret_key
S3_ENDPOINT=https://ams3.digitaloceanspaces.com
S3_REGION=ams3
S3_BUCKET=your_bucket_name

# MongoDB (Optional - for results storage)
MONGO_CONNECTION_STRING=mongodb://username:password@localhost:27017/dbname
MONGO_DATABASE_NAME=d_data

# Logging
LOG_LEVEL=INFO
```

### Data Sources
The pipeline expects input data in the following structure:
- **Financial data**: `model-data/input/financials.csv`
- **Emission scopes**: `model-data/input/scopes.csv`
- **Categorical features**: `model-data/input/categoricals.csv`
- **Statistical features**: `model-data/input/statisticals.csv`

### Output Artifacts
- **Models**: Serialized pipeline objects in `model-data/output/`
- **Reports**: Performance metrics and evaluation results in `local/prod_runs/`
- **Remote storage**: Optional S3 backup of all artifacts

## Architecture

### Core Components
- **Data Management**: Unified interface for multiple data sources with caching
- **Preprocessing**: Financial transformations, categorical encoding, and feature scaling
- **Feature Selection**: VIF-based filtering and dimensionality reduction
- **Model Training**: Multi-scope estimators with hyperparameter optimization
- **Post-processing**: Confidence intervals, missing value imputation, and explainability

### Pipeline Design
The system implements a modular pipeline architecture where each component can be independently configured and optimized. The default pipeline includes:

1. **Data preprocessing** with financial ratio calculations
2. **Feature engineering** and selection
3. **Multi-scope model training** using ensemble methods
4. **Confidence estimation** for prediction uncertainty
5. **Model persistence** and evaluation reporting

## Usage Examples

### Training Pipeline
```python
from pipeline.core import DefaultPipeline
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator

pipeline = DefaultPipeline(
    scope_estimator=EvenWeightMiniModelArmyEstimator(10, n_trials=40),
    # Additional configuration...
)
pipeline.optimise(X_train, y_train).fit(X_train, y_train)
```

### Inference
```python
# Load trained model
model = load_model_from_disk('model-data/output/model.pkl')

# Make predictions
predictions = model.predict(features)
```

## Performance

The system has been extensively evaluated across multiple datasets and configurations, achieving:
- **High accuracy** across different emission scopes
- **Robust generalization** to unseen company data
- **Efficient training** with optimized hyperparameter search
- **Scalable inference** for production workloads

## Dependencies

### Core ML Libraries
- scikit-learn: Machine learning algorithms and utilities
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Visualization and plotting

### Specialized Components
- pygraphviz: Graph visualization (optional)
- boto3: AWS S3 integration
- pymongo: MongoDB connectivity
- optuna: Hyperparameter optimization

## Development

### Project Structure
```
architectura/
├── base/           # Core abstractions and utilities
├── pipeline/       # ML pipeline implementation
├── scope_estimators/ # Emission prediction models
├── preprocessors/  # Feature engineering components
├── imputers/       # Missing value handling
├── experiments/    # Research and evaluation scripts
├── notebooks/      # Analysis and exploration
└── tests/          # Unit and integration tests
```

### Contributing
The codebase follows standard Python practices with:
- Type hints for better code clarity
- Comprehensive error handling
- Extensive logging and monitoring
- Modular design for easy extension

## License

This project is proprietary software. All rights reserved.

---

For detailed technical specifications and methodology, refer to the comprehensive technical report: `Portfolio__Carbon_Emission_Modelling.pdf`
