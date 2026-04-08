from hmmlearn.hmm import GaussianHMM

def train_hmm(features):
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
    model.fit(features)
    return model

def predict_regime(model, features):
    return model.predict(features)