# RAG

## Overview

Retrieval-Augmented Generation (RAG) es una técnica en inteligencia artificial que combina recuperación de información (retrieval) y generación de texto (generation) para mejorar la calidad y precisión de las respuestas de un modelo de lenguaje.

---

## Ejecución python
Podemos encontrar un script de python donde se contienen las instrucciones de ejecución. Deben reemplazarse las llaves por valores de API KEY de openAI y pinecode para poderlo ejecutar. Luego de iniciar la ejecución este va a pedir una entrada a lo que va a dar una posterior respuesta.

## Arquitectura

1. **Carga de documentos**: Carga documentos desde una fuente web En este caso usamos https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
2. **División de texto**: Divide los documentos en fragmentos más pequeños para una recuperación eficiente.  
3. **Almacén de vectores**: Almacena las incrustaciones de los documentos para búsqueda por similitud.  
4. **Recuperación**: Recupera los fragmentos de documentos relevantes según las consultas del usuario.  
5. **Generación**: Usa un modelo de lenguaje para generar respuestas basadas en el contenido recuperado.

---

## Instalación

### Requisitos

- Python 3.12
- OpenAI API key.
- Pinecone API key

### Instalación

1. **Instalación de Librerías Requeridas**:
Antes de comenzar, instala las librerías necesarias con pip:
    
    ```bash
    pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph beautifulsoup4 langchain-openai langchain-pinecone pinecone-notebooks
    
    ```
    
2. Configuración de Claves API
Para utilizar OpenAI y Pinecone, es necesario establecer sus claves de API como variables de entorno.
    
    ### OpenAI API Key:
    
    ```python
    import getpass
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    
    ```
    
    ### Pinecone API Key:
    
    ```python
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    ```
    
3. Creación de un Índice en Pinecone
Se creará un índice en Pinecone para almacenar las representaciones vectoriales de los documentos.
    
    ```python
    from pinecone import Pinecone, ServerlessSpec
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "langchain-test-index"  # Change if desired
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    index = pc.Index(index_name)
    
    ```
    
4. Inicialización del Modelo de Embeddings
Para codificar los documentos en vectores, se utilizará el modelo de embeddings de OpenAI.
    
    ```python
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    ```
    
5.  Configuración del Almacén de Vectores
Pinecone se utilizará como la base de datos para almacenar y recuperar documentos relevantes.
    
    ```python
    from langchain_pinecone import PineconeVectorStore
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    ```
    

---

## Ejemplo de las salidas

### Entrada:

```python
result = graph.invoke({"question": "What is Task Decomposition?"})
print(result["answer"])

```

### Salida:

```
Task Decomposition is the process of breaking down a complicated task into smaller, manageable steps. It involves techniques like Chain of Thought (CoT), where the model is prompted to "think step by step," and can include various methods such as simple prompting, task-specific instructions, or human inputs.

```

---

## Capturas

![image]()

![image]()

---


* **Johan Alejandro Estrada Pastran** 
