import orderclassifier as model
import preprocess as preprocess

if __name__ == "__main__":
    print()
    data = preprocess.read_data('data/train.csv')
    clf = model.read_model('models/rf_total.model')

    model.eval_model(clf, data)

