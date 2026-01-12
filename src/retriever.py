import numpy as np
import faiss
import pickle
import os

class MedicalRetriever:
    def __init__(self, data_path="data/embeddings"):
        self.index_path = os.path.join(data_path, "faiss_index.bin")
        self.meta_path = os.path.join(data_path, "chunks_metadata.pkl")
        self.embed_path = os.path.join(data_path, "embeddings.npy")

        # Load resources safely
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.embeddings = np.load(self.embed_path)
        except Exception as e:
            print(f"Error loading resources: {e}")
            print("Ensure you have run the data pipeline in the notebook first.")

    def search(self, query_embedding, top_k=5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results = []

        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    "score": float(score),
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"]
                })
        return results