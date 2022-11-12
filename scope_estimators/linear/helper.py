from sklearn.preprocessing import PolynomialFeatures

class PolynomialFeaturesMixin:
    def __init__(self, degree=3, **kwargs) -> None:
        self.polynomializer = PolynomialFeatures(degree=degree)
        super().__init__(**kwargs)
        
    def prepare_features(self, X):
        X = self.polynomializer.fit_transform(X)
        return X
    
