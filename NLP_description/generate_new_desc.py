import prepare_data
import generate_desc
from pickle import load
from keras.models import load_model

tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 33
# load the model
model = load_model('model_0.h5')
# load and prepare the photograph
photo = prepare_data.extract_features_filename('../dataset/FlickerDenoisedImages/Flicker_dn_drunet_color/10815824_2997e03d76.jpg')
# generate description
description = generate_desc.generate_desc(model, tokenizer, photo, max_length)
print(description)