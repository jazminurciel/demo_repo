# -*- coding: utf-8-sig -*-
from flask import abort, Flask, jsonify, request
from symspellpy.symspellpy import SymSpell, Verbosity
from time import time
import pickle
import spacy
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

model = pickle.load(open('models/TF-IDF_MNB.pkl','rb'))
nlp = spacy.load("es_core_news_md")

sym_spell = SymSpell(
    max_dictionary_edit_distance=3,
    prefix_length=7,
    count_threshold=1,
    compact_level=5,
)

sym_spell.load_dictionary(corpus='resources/es_real_freq_full.txt',term_index=0,count_index=1,encoding='utf-8')

print("Everything is load")


@app.route('/api/analyzeSentiment', methods=['POST'])
def analyzeSentiment():
    text = request.form.get('text')
    start = time()
    print(f"\nInput text: {text}")
    clean = clean_text(text)
    score = model.predict_proba(clean)[0][1]
    print(f"Score: {score}")
    print(f"Process took {time()-start} seconds to finish\n")

    response = {'Score':str(score)}

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
