# Asistente RAG UFRO

Proyecto de asistente conversacional con **RAG (Retrieval-Augmented Generation)** para responder preguntas sobre la normativa oficial de la Universidad de La Frontera (UFRO).  
Implementado en **Python** con **FAISS**, **sentence-transformers**, **ChatGPT** y **DeepSeek**, y desplegable en **AWS Ubuntu + Flask**.

---

## 📌 Instalación y Uso

Ejecuta los siguientes comandos desde la raíz del proyecto:

```bash
# Crear entorno virtual:
python -m venv env

# Activar entorno virtual:
.\env\Scripts\Activate.ps1

# Instalar dependencias:
pip install -r requirements.txt

# Crear archivo .env en la raíz del proyecto con tus API keys
# Generar embeddings e índice FAISS:
python chunk_embed.py ingest --data_dir data/docs --out_dir index --chunk_size 400 --overlap 70
python build_faiss.py

# Probar por CLI:
python app.py "¿Cuándo inician las clases según el calendario académico 2025?" --provider chatgpt

# Ejecutar la interfaz Flask:
python app_flask.py

## 📌 Política de Ética y Abstención

Este asistente **NO inventa respuestas**.  
Cuando la información no se encuentra en los documentos oficiales indexados, responde explícitamente:

> `"No encontrado en normativa UFRO"`

De esta forma se evita la generación de **alucinaciones** y se garantiza la confiabilidad de la respuesta.

---

## 📌 Vigencia Normativa

Los documentos utilizados corresponden a normativa oficial vigente de la UFRO, descargados el **29 de septiembre de 2025**.  
La vigencia se debe actualizar de forma periódica para asegurar que las respuestas correspondan a la versión más reciente.

---

## 📌 Privacidad

- El sistema **no almacena información personal de usuarios**.  
- Solo procesa las preguntas y devuelve respuestas con base en documentos normativos.  
- Se recomienda que su uso esté limitado a propósitos académicos e institucionales.  
- Los modelos (ChatGPT / DeepSeek) pueden reflejar **sesgos propios de su entrenamiento**.

---

## 📌 Tabla de Trazabilidad de Documentos

| doc_id | Documento                           | Fuente (URL)                                   | Vigencia |
|--------|--------------------------------------|-----------------------------------------------|----------|
| D1     | Calendario Académico 2025           | [UFRO - Calendario Académico](https://www.ufro.cl/calendario-academico) | Año 2025 |
| D2     | Reglamento de Régimen de Estudios   | Documento PDF oficial UFRO                     | 2023     |
| D3     | Reglamento de Convivencia Universitaria | Documento PDF oficial UFRO                 | 2023     |

---
