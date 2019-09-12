# -*- coding: utf-8-sig -*-
from flask import abort, Flask, jsonify, request
from symspellpy.symspellpy import SymSpell, Verbosity
from time import time
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
import pickle
import re


def clean_text(text):

    text = text.encode('latin', 'ignore').decode('latin')

    max_edit_distance_lookup = 3
    text = sym_spell.lookup_compound(text,max_edit_distance_lookup)[0].term

    tokens= nlp(u""+text)
    new_text= ' '.join([t.lemma_ for t in tokens])

    return [new_text]

print("Initializing variables")

app = Flask(__name__)

model = load_model('NN_models/FastText_1CNN-BGRU_lemma.h5')
tokenizer = pickle.load(open('models/tokenizer_24_embedding_fasttext_1cnn-bgru.pkl','rb'))
max_len=24

nlp = spacy.load("es_core_news_md")

sym_spell = SymSpell(
    max_dictionary_edit_distance=3,
    prefix_length=7,
    count_threshold=1,
    compact_level=5,
)

sym_spell.load_dictionary(corpus='resources/es_real_freq_full.txt',term_index=0,count_index=1,encoding='utf-8')

print("Everything is loaded")

@app.route('/api_nlp/analyzeSentiment', methods=['POST'])
def analyzeSentiment():
    text = request.form.get('text')#obtiene el texto
    start = time()#toma tiempo(pidio el dr juan)
    print(f"\nInput text: {text}")
    clean = clean_text(text)#limpia el texto
    print(f"Text after preprocessing: {clean[0]}\n")
    text_tokens = tokenizer.texts_to_sequences(clean)
    text_pad = pad_sequences(text_tokens,maxlen=max_len)
    score = model.predict(text_pad)[0][0]
    print(f"Score {score}")
    print(f"\nProcess took {time()-start} seconds to finish\n")

    response = {'Score':str(score)}

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1')
