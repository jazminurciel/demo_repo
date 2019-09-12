# importar dependencias
from flask import Flask, request, make_response, jsonify, render_template, abort
import dialogflow
import requests #modulo de peticiones de python
import json
import pusher
from symspellpy.symspellpy import SymSpell, Verbosity
from time import time
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
import pickle
import re
# sqlite3
#from sqlite3 import Error



#def clean_text(text):

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

#@app.route('/api_nlp/analyzeSentiment', methods=['POST'])
#def analyzeSentiment(mydata):

    start = time()#toma tiempo(pidio el dr juan)
    print(f"\nInput text: {mydata}")
    clean = clean_text(mydata)#limpia el texto
    print(f"Text after preprocessing: {clean[0]}\n")
    text_tokens = tokenizer.texts_to_sequences(clean)
    text_pad = pad_sequences(text_tokens,maxlen=max_len)
    score = model.predict(text_pad)[0][0]
    print(f"Score {score}")
    print(f"\nProcess took {time()-start} seconds to finish\n")

    return score

def results():
	req = request.get_json(force=True) # En la variable req se guardara la peticion que devolvera datos en formato json, extraidos de dialogflow.
	action = req.get('queryResult').get('action') # De los datos obtenidos en la variable req, se va a extraer 'action', esta nos ayuda a buscar el intent que necesitamos.
	result = {} # Se guarda un diccionario vacio en la variable result.
	
	if action=="pregunta.action":# Si el intent en el apartado de "action and parameters" de dialogflow es igual a "action1.astros" entonces entra y se siguen ejecutando las siguientes lineas de código.
		val = req.get('queryResult').get('parameters')# De los datos obtenidos en la variable req, le pedimos qu nos mande todos los parametros que obtiene de dicho intent.
		datos=val.get('sentimiento')# En la variable astros guardaremos las entidades obtenidas, de los parametros obtenidos de la variable val.
		score1=analyzeSentiment(datos)
		if score1>=0.5:
			res1="positivo"
		else:
			res1="negativo"

		result["fulfillmentText"] = res1# Enviamos el concepto y la oración.
		result = jsonify(result)# Convertimos lo obtenido en result a formato json para que lo entienda dialogflow.
		return make_response(result)# Manda result a dialogflow.

	if action=="nombreyedad.action":
		val = req.get('queryResult').get('parameters')# De los datos obtenidos en la variable req, le pedimos qu nos mande todos los parametros que obtiene de dicho intent.
		nombre=val.get('nombre')
		edad=val.get('edad')
		conn = sqlite3.connect("mydb.db")
		cursor = conn.cursor()
		#cursor.execute("SELECT * FROM Cuarto")
		cursor.execute('''INSERT INTO estudiantes (nombre, edad) VALUES (?,?)''',(nombre,edad))
		conn.commit()
		conn.close()
		resp='la informacion se agrego con exito a la BD'
		result["fulfillmentText"] = resp# Enviamos el concepto y la oración.
		result = jsonify(result)# Convertimos lo obtenido en result a formato json para que lo entienda dialogflow.
		return make_response(result)#


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
	# devolver respuesta
	return results()

# correr la app
if __name__ == '__main__':
	app.run()