# diagnosis.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class HealthcareDiagnoser:
    def __init__(self, faiss_folder="faiss_index"):
        self.faiss_folder = faiss_folder
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.load_faiss_index()
    
    def load_faiss_index(self):
        """Load FAISS index and metadata"""
        if not os.path.exists(self.faiss_folder):
            raise FileNotFoundError(f"FAISS folder '{self.faiss_folder}' not found. Run create_faiss.py first!")
        
        index_path = os.path.join(self.faiss_folder, "index.faiss")
        docs_path = os.path.join(self.faiss_folder, "docs.pkl")
        metadata_path = os.path.join(self.faiss_folder, "metadata.pkl")
        
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"✅ Loaded {len(self.metadata)} disease entries")
    
    def get_complete_disease_info(self, disease_name):
        """Retrieve complete disease information from metadata"""
        for meta in self.metadata:
            if meta['disease'].lower() == disease_name.lower():
                return meta
        return None
    
    def search_similar_diseases(self, symptoms_text, top_k=1):
        """Search for similar diseases using RAG"""
        query_embedding = self.embedder.encode([symptoms_text])
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                disease_meta = self.metadata[idx]
                results.append({
                    'disease': disease_meta['disease'],
                    'symptoms': disease_meta['symptoms'],
                    'confidence': max(0, 100 - (distances[0][i] * 20)),  # Convert to percentage
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def diagnose(self, symptoms_text, top_k=15):
        """Diagnose diseases and return complete information"""
        # RAG: Retrieve similar diseases
        similar_diseases = self.search_similar_diseases(symptoms_text, top_k)
        
        # Remove duplicates and sort by confidence
        unique_diseases = {}
        for disease_info in similar_diseases:
            disease_name = disease_info['disease']
            if disease_name not in unique_diseases or disease_info['confidence'] > unique_diseases[disease_name]['confidence']:
                unique_diseases[disease_name] = disease_info
        
        # Sort by confidence
        sorted_diseases = sorted(
            unique_diseases.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        # Get complete information for top diseases
        final_diagnoses = []
        for disease_info in sorted_diseases[:3]:  # Top 3 diseases
            complete_info = self.get_complete_disease_info(disease_info['disease'])
            if complete_info:
                final_diagnoses.append({
                    'disease': complete_info['disease'],
                    'symptoms': complete_info['symptoms'],
                    'medicines': complete_info['medicines'],
                    'precautions': complete_info['precautions'],
                    'confidence': min(95, disease_info['confidence'])  # Cap at 95% for safety
                })
        
        return final_diagnoses

    def generate_response(self, symptoms_text):
        """Generate complete medical response with all information"""
        diagnoses = self.diagnose(symptoms_text)
        
        if not diagnoses:
            return {
                'diagnoses': [],
                'message': "I couldn't find a matching condition for your symptoms. Please consult a healthcare professional for accurate diagnosis."
            }
        
        return {
            'diagnoses': diagnoses,
            'message': f"Found {len(diagnoses)} possible conditions matching your symptoms."
        }

# Test function
def test_diagnosis():
    """Test the diagnosis system with complete row retrieval"""
    try:
        print("🧪 Testing Healthcare Diagnosis System...")
        diagnoser = HealthcareDiagnoser()
        
        # Test cases
        test_symptoms = [
            "fever and cough",
            "headache and nausea",
            "chest pain and shortness of breath",
            "joint pain and fatigue"
        ]
        
        for symptoms in test_symptoms:
            print(f"\n🔍 Symptoms: {symptoms}")
            print("-" * 60)
            
            response = diagnoser.generate_response(symptoms)
            
            if response['diagnoses']:
                print(f"📊 {response['message']}")
                print()
                
                for i, diagnosis in enumerate(response['diagnoses'], 1):
                    print(f"🏥 {i}. {diagnosis['disease']} ({diagnosis['confidence']:.1f}% confidence)")
                    print(f"   🤒 Symptoms: {', '.join(diagnosis['symptoms'][:5])}")
                    print(f"   💊 Medicines: {', '.join(diagnosis['medicines'][:3])}")
                    print(f"   🛡️  Precautions: {', '.join(diagnosis['precautions'][:3])}")
                    print()
            else:
                print(response['message'])
                
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please run 'python src/create_faiss.py' first to create the FAISS index!")

if __name__ == "__main__":
    test_diagnosis()