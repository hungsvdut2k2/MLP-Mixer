from sklearn.preprocessing import LabelEncoder
from models.mlp_mixer import MlpMixer


def convert_category_to_label(categories: list) -> list:
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(categories)
    return encoded_categories
