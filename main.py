import streamlit as st
from pymongo import MongoClient
from azure.cosmos import CosmosClient
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración de Cosmos DB
cosmos_endpoint = os.environ['COSMOS_ENDPOINT']
cosmos_key = os.environ['COSMOS_KEY']
cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)

# Base de datos para usuarios (SQL API)
user_database = cosmos_client.get_database_client("user_database")
user_container = user_database.get_container_client("users")

# Configuración de Cosmos DB (API MongoDB) para vectores
mongo_connection_string = os.environ['MONGODB_CONNECTION_STRING']
mongo_client = MongoClient(mongo_connection_string)
vector_db = mongo_client['aideatext_db']
vector_collection = vector_db['text_vectors']

# Inicializar vectorizador
vectorizer = TfidfVectorizer()

def test_connection():
    try:
        # Probar conexión SQL API
        user_container.read()
        # Probar conexión MongoDB API
        vector_db.command('ping')
        return True
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return False

def register_user(username, email):
    try:
        user_container.create_item({
            'id': username,
            'email': email
        })
        return True
    except Exception as e:
        st.error(f"Error al registrar usuario: {e}")
        return False

def process_and_store_text(username, text):
    try:
        # Convertir texto a vector
        vector = vectorizer.fit_transform([text]).toarray()[0]
        
        # Almacenar en la base de datos vectorial
        vector_collection.insert_one({
            'username': username,
            'text': text,
            'vector': vector.tolist()
        })
        return True
    except Exception as e:
        st.error(f"Error al procesar y almacenar texto: {e}")
        return False

# Interfaz de Streamlit
st.title("AIdeaText")

# Probar conexión
if st.button("Probar Conexión"):
    if test_connection():
        st.success("Conexión exitosa a las bases de datos")
    else:
        st.error("Error de conexión")

# Registro de usuario
st.header("Registro de Usuario")
username = st.text_input("Nombre de usuario")
email = st.text_input("Correo electrónico")
if st.button("Registrar Usuario"):
    if register_user(username, email):
        st.success("Usuario registrado con éxito")

# Procesamiento de texto
st.header("Procesar y Almacenar Texto")
text_input = st.text_area("Ingrese su texto aquí")
if st.button("Procesar y Almacenar"):
    if process_and_store_text(username, text_input):
        st.success("Texto procesado y almacenado con éxito")

# Aquí puedes añadir más funcionalidades de tu aplicación
