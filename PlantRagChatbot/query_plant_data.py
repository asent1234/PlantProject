import requests
import json
from dotenv import load_dotenv
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the project root to the Python path
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from root .env file
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
# from langchain_chrPlant import ChrPlant
# from create_database import OllamaEmbeddings

class PlantQueryEngine:
    CHROMA_PATH = "chrPlant"
    
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
   
    {context}
   
    ---
    Answer the question based on the above context primarily, but take into account that the context may contain information
    regarding larger crop and agricultural systems. Reduce the application and apply solutions for singular pot
    plants, so things like crop rotation cannot be applied. Make a determination of which ailment it is specifically
    and focus on that, and answer in a paragraph form, do not bring back in the percentage numbers: 
   
    {question}
    """

    DIRECT_PROMPT_TEMPLATE = """
    You are an expert in plant plant care and diseases. Please answer the following question about plant plants. 
    Focus on practical solutions for home gardeners growing plantes in pots or small gardens. 
    If the question is about a disease or problem, explain how to identify it and provide specific treatment steps: 

    {question}
    """

    def __init__(self, model_name="deepseek-r1:14b", similarity_threshold=0.7, max_results=12):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        # self.embeddings = OllamaEmbeddings(model_name=model_name)
        # self.db = ChrPlant(persist_directory=self.CHROMA_PATH, embedding_function=self.embeddings)

    def get_deepseek_completion(self, prompt, stream=False):
        """
        Query the DeepSeek API for a completion.
        """
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        endpoint = "https://api.deepseek.com/v1/chat/completions"
        try:
            if stream:
                response = requests.post(endpoint, headers=headers, json=data, stream=True)
                response.raise_for_status()
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[6:]
                            try:
                                json_response = json.loads(line)
                            except Exception as e:
                                print("[Streaming] Could not parse line as JSON:", line)
                                continue
                            # DeepSeek streaming returns 'choices' with 'delta' and 'content'
                            if 'choices' in json_response and json_response['choices']:
                                delta = json_response['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    yield content
                return stream_generator()
            else:
                response = requests.post(endpoint, headers=headers, json=data)
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    print("DeepSeek API Error Response:", response.text)
                    raise
                try:
                    result = response.json()
                except Exception as e:
                    print("[Non-streaming] Could not parse response as JSON:", response.text)
                    raise
                # DeepSeek non-streaming returns 'choices' with 'message' and 'content'
                if 'choices' in result and result['choices']:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content')
                    if content:
                        return content
                return "[No valid response from DeepSeek]"
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to DeepSeek: {str(e)}")


    # Streaming handled in get_deepseek_completion; no separate function needed.

    def search_similar_documents(self, query_text):
        results = self.db.similarity_search_with_relevance_scores(query_text, k=self.max_results)
        if len(results) == 0 or results[0][1] < self.similarity_threshold:
            return None
        return results

    def format_context(self, results):
        return "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    def query(self, query_text, force_direct=False, stream=False):
        try:
            results = None if force_direct else self.search_similar_documents(query_text)
            
            if force_direct or results is None:
                direct_prompt = self.DIRECT_PROMPT_TEMPLATE.format(question=query_text)
                return self.get_deepseek_completion(direct_prompt, stream=stream)
            
            context_text = self.format_context(results)
            rag_prompt = self.PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            return self.get_deepseek_completion(rag_prompt, stream=stream)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    engine = PlantQueryEngine()
    print("PlantQueryEngine initialized and ready to use.")
