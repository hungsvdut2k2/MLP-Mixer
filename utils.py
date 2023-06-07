from sklearn.preprocessing import LabelEncoder


def convert_category_to_label(categories: list):
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(categories)
    return encoded_categories
