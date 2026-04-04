from Pipeline.feature_extractor import FeatureExtractor


def extract_features(events):
    extractor = FeatureExtractor()
    return extractor.extract_features_from_events(events)