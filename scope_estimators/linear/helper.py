from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


class PolynomialFeaturesMixin:
    # TODO: Introduce a feature transformer that adds non-polynomial features
    def __init__(self, degree=3, **kwargs) -> None:
        self.polynomializer = PolynomialFeatures(degree=degree, include_bias=False)
        super().__init__(**kwargs)
        
    def prepare_features(self, X):
        X = self.polynomializer.fit_transform(X)
        return X

class NormalizedFeaturesMixin:
    def __init__(self, **kwargs) -> None:
        self.normalizer = MinMaxScaler(**kwargs)
        super().__init__(**kwargs)
        
    def prepare_features(self, X):
        X = self.normalizer.fit_transform(X)
        return X
    
