### Ghosts as Unstructured Data: A Study Guide

### Quiz

#### Instructions: Answer the following questions in 2-3 sentences each.

* Why are ghosts considered unstructured data?
* What are the advantages of using Milvus for storing and searching ghost data?
* What is the purpose of adding a "ghostclass" partition key to the schema?
* What are some examples of metadata that can be used to describe ghosts?
* Why is it important to index fields in a Milvus collection?
* What is the role of LLMs in generating descriptions for ghost images?
* What are some examples of embedding models used in the Ghost Capture project?
* How can CLIP be used to search for ghost images?
* What is hybrid search and why is it powerful for searching ghost data?
* What are some future applications planned for the Ghost Capture project?

#### Answer Key

````
Ghosts are considered unstructured data because they lack a physical form, structure, or material body. Their attributes and manifestations are often subjective and difficult to categorize in a traditional database.

Milvus is advantageous for ghost data due to its ability to handle multiple types of similarity searches, including text, image, audio, and video. It allows for hybrid and multimodal searches and can handle large-scale datasets efficiently.

Adding a "ghostclass" partition key improves performance by distributing data across eight predefined categories of ghosts. This enables faster and more targeted searches based on ghost classification.

Metadata examples include ghost classification (Class I-VII), physical descriptions, sighting timestamps, location details (latitude, longitude, country, zip code), and historical references or folklore associated with the ghost.

Indexing fields significantly improves search speed and efficiency. Different indexing methods, such as the Inverted Index used for the "ghostclass" and zip code fields, optimize data retrieval for specific queries.

LLMs like "mistral-nemo" and Llava are used to automatically generate detailed descriptions for ghost images. This is particularly useful for bulk loading data where manual description creation is impractical.

Embedding models include BGE-M3, SPLADE, CLIP, and Alibaba-NLP/gte-base-en-v1.5. These models convert different data types (text, images) into numerical vectors, allowing for similarity comparisons.

CLIP allows for searching ghost images using both text and images as queries. The model embeds both text and images into the same vector space, enabling cross-modal retrieval based on semantic similarity.

Hybrid search combines multiple search techniques, including vector similarity search, scalar filtering, and multimodal search. This enables complex queries, such as finding ghosts matching specific descriptions, locations, and image features.

Future applications include a ghost reporting app, a thermal camera app on a Raspberry Pi for detection, 
a collector trap also on a Raspberry Pi, a RAG (Retrieval Augmented Generation) app, a Java enterprise application for data management, and advanced analytics using Jupyter notebooks.
````

#### Essay Questions

````
Discuss the challenges of representing and storing unstructured data, using ghosts as an example. How can techniques like vector embeddings and multimodal models address these challenges?

Explain the concept of hybrid search and its relevance to the Ghost Capture project. Describe a scenario where hybrid search would be particularly beneficial for finding specific ghost data.

Evaluate the ethical implications of capturing and analyzing ghost data. Consider issues of privacy, consent, and the potential misuse of such information.

Analyze the role of community involvement in the Ghost Capture project. How can citizen science and crowdsourced data contribute to the understanding of paranormal phenomena?

Imagine a future where ghost data is widely available and integrated with other datasets. Discuss the potential societal, cultural, and scientific impacts of such a development.

````


### Glossary of Key Terms

#### Term/Definition

````
Unstructured Data
Data that does not have a predefined format or organization, such as text, images, audio, and video.

Milvus
An open-source vector database designed for storing, indexing, and searching large-scale vector data.

Schema
A blueprint that defines the structure and organization of data in a database.

Partition Key
A field used to divide data into smaller, more manageable chunks for improved performance and scalability.

Metadata
Data that provides information about other data, such as descriptions, timestamps, and location details.

Indexing
The process of creating data structures that allow for efficient searching and retrieval of data.

LLM (Large Language Model)
A type of artificial intelligence model trained on a massive text dataset, capable of generating text, translating languages, and answering questions in a comprehensive and informative way.

Embedding Model
A model that converts data into numerical vectors, representing the data's meaning and relationships.

CLIP (Contrastive Language-Image Pre-training)
A model that learns to connect images and text, allowing for cross-modal search and retrieval.

Hybrid Search
A search strategy that combines different search techniques, such as vector similarity search, scalar filtering, and multimodal search, to find the most relevant results.

RAG (Retrieval Augmented Generation)A technique that enhances text generation by retrieving relevant information from a knowledge base, improving the accuracy and factual grounding of generated content.

BLOB (Binary Large Object)
A data type used to store large amounts of binary data, such as images, audio files, and video files, in a database.

GIS (Geographic Information System)
A system designed to capture, store, analyze, manage, and present spatial or geographic data.

Scalar Filtering
A search technique that uses structured data (e.g., dates, numbers, categories) to narrow down search results.

Multimodal Search
A search technique that combines different data modalities, such as text, images, and audio, to find relevant results.

Dense Vector
A vector representation of data where most elements have non-zero values. Often used for embedding models like BGE-M3.

Sparse Vector
A vector representation of data where most elements are zero. Commonly used for text data embedding with models like SPLADE.

Trie
A specialized tree-like data structure used for efficient string prefix searching. It is helpful for indexing data like the "ghostclass" category.

Inverted Index
An index that maps words or terms to the documents or records where they occur. It is frequently used for text search and retrieval, and in this case, for searching zipcodes.

````



