import prepare_data
import generate_desc
from pickle import load
from keras.models import load_model

tokenizer = load(open('Pickle/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep001-loss4.712-val_loss3.999.h5')
# load and prepare the photograph
photo = prepare_data.extract_features_filename('example2.jpg')
# generate description
description = generate_desc.generate_desc(model, tokenizer, photo, max_length)
print(description)
