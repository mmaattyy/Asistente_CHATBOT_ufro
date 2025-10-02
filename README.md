# Asistente RAG UFRO

Proyecto de asistente conversacional con **RAG (Retrieval-Augmented Generation)** para responder preguntas sobre la normativa oficial de la Universidad de La Frontera (UFRO).  
Implementado en **Python** con **FAISS**, **sentence-transformers**, **ChatGPT** y **DeepSeek**, y desplegable en **AWS Ubuntu + Flask**.

---

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

## 📌 Ejemplo de Uso CLI

```bash
python app.py "¿Cuándo inician las clases según el calendario académico 2025?" --provider chatgpt
