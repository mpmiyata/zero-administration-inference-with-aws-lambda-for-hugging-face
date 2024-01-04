# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_url
import torch
import pickle
import requests 

MODEL_PATH = 'mpmiyata/ds_bertimbau_sim_BP'
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def download_label_encoder(repo_id, filename):
    url = hf_hub_url(repo_id, filename)

    # Definir o caminho de destino no diretório /tmp
    tmp_path = f'/tmp/{filename}'

    # Fazer o download do arquivo e salvar em /tmp
    response = requests.get(url)
    if response.status_code == 200:
        with open(tmp_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Erro ao baixar o arquivo: {response.status_code}")

    # Carregar o label encoder
    with open(tmp_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return label_encoder

label_encoder = download_label_encoder(repo_id=MODEL_PATH, filename='label_encoder.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    prediction_number = predictions.cpu().numpy()[0]
    predicted_label_text = label_encoder.inverse_transform([prediction_number])[0]

    return predicted_label_text

def handler(event, context):
    response = {
        "statusCode": 200,
        "body": predict(event['text'])
    }
    return response