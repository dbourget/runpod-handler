
from sentence_transformers import SentenceTransformer
import runpod
#from pprint import pprint

#use cuda unless env var CPU_ONLY is set to true
device_to_use = 'cuda' if not runpod.env.get('CPU_ONLY') else 'cpu'

model = SentenceTransformer('dbourget/philai-v1', device=device_to_use)

def embeddings(job):
    
    sentences = job['input']['input']
    embeddings = model.encode(sentences)
    # return embeddings in this format: data: [ {embedding: [first embedding]}, {embedding: [second embdding]} ]
    return { 'usage': { 'total_tokens': 0 }, 'data': list(map(lambda x: { 'embedding': x.tolist() }, embeddings)) }


#print(embeddings({'input': ['Hello World!',"How are you?"]}))
runpod.serverless.start({"handler": embeddings}) 