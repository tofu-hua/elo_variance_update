from sklearn.preprocessing import LabelEncoder
def encode_marks(marks):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(marks)
    oh = np.zeros((len(marks), len(encoder.classes_)))

    oh[np.arange(len(marks)), encoded] = 1

    return oh, encoder.classes_