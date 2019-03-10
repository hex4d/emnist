from main import *
import collections

def test(test_gen, model):
    incorrect_count = []
    count = 0
    for i in range(len(test_gen)):
        x, y = next(test_gen)
        pred = model.predict(x)
        for i in range(len(x)):
            if y[i].argmax() != pred[i].argmax():
                key = category_to_char(pred[i].argmax()) + '-' + category_to_char(y[i].argmax())
                incorrect_count.append(key)
                if key == 'p-d' and count < 10:
                    count = count + 1
                    plt.imshow(x[i].reshape((28, 28)))
                    plt.show()
    return collections.Counter(incorrect_count)


filename = ''
model_wrapper = ModelV3()
model = models.load_model(os.path.join(model_wrapper.model_path(model_base_dir), filename))

pred = model.predict_generator(test_gen, steps=len(test_gen))

train_gen, val_gen, test_gen = load_data()
