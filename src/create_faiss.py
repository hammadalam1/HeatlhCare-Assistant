import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSCreator:
    def __init__(self):
        self.health_file = "data/health.json"
        self.faiss_folder = "faiss_index"
        self.embed_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        
    def check_file(self):
        """Check if the health file exists and is readable"""
        print("🔍 Checking health file...")
        
        if os.path.exists(self.health_file):
            file_size = os.path.getsize(self.health_file)
            print(f"✅ Health file found: {self.health_file} ({file_size} bytes)")
            
            # Read and validate JSON structure
            try:
                with open(self.health_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'diseases' in data and isinstance(data['diseases'], list):
                    print(f"✅ Valid JSON structure with {len(data['diseases'])} diseases")
                    # Show first few diseases
                    for i, disease in enumerate(data['diseases'][:3], 1):
                        print(f"   {i}. {disease.get('name', 'Unknown')}")
                    return True
                else:
                    print("❌ Invalid JSON structure: missing 'diseases' array")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON format: {e}")
                return False
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return False
        else:
            print(f"❌ Health file not found: {self.health_file}")
            return False
    
    def parse_health_file(self):
        """Parse the JSON health file"""
        diseases = []
        
        print(f"\n📖 Parsing health file: {self.health_file}")
        
        try:
            with open(self.health_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'diseases' not in data:
                print("❌ JSON file missing 'diseases' key")
                return []
            
            for disease_info in data['diseases']:
                disease = disease_info.get('name', '').strip()
                symptoms = disease_info.get('symptoms', [])
                medicines = disease_info.get('medicines', [])
                precautions = disease_info.get('precautions', [])
                
                if disease:
                    disease_data = {
                        'disease': disease,
                        'symptoms': symptoms,
                        'medicines': medicines,
                        'precautions': precautions
                    }
                    diseases.append(disease_data)
            
            print(f"✅ Successfully parsed {len(diseases)} diseases")
            
            # Show first 3 parsed entries
            for i, disease_data in enumerate(diseases[:3], 1):
                print(f"   📝 Disease {i}: {disease_data['disease']}")
                print(f"      Symptoms: {disease_data['symptoms'][:3]}")
                print(f"      Medicines: {disease_data['medicines'][:2]}")
                print(f"      Precautions: {disease_data['precautions'][:2]}")
            
        except Exception as e:
            print(f"❌ Error parsing {self.health_file}: {e}")
            import traceback
            traceback.print_exc()
        
        return diseases
    
    def create_knowledge_base(self):
        """Create comprehensive knowledge base from the health file"""
        print("\n" + "="*50)
        print("CREATING KNOWLEDGE BASE FROM HEALTH FILE")
        print("="*50)
        
        # First check if file exists
        if not self.check_file():
            raise Exception("Health file not found or unreadable!")
        
        # Parse the health file
        disease_data = self.parse_health_file()
        
        documents = []
        metadata = []
        
        print(f"\n🔍 Found {len(disease_data)} diseases")
        
        if not disease_data:
            print("❌ No diseases found in the health file!")
            return [], []
        
        for disease_info in disease_data:
            disease = disease_info['disease']
            symptoms = disease_info['symptoms']
            medicines = disease_info['medicines']
            precautions = disease_info['precautions']
            
            # Create multiple document variations for better retrieval
            doc_variations = []
            
            # Variation 1: Comprehensive document
            doc_parts = [f"DISEASE: {disease}"]
            if symptoms:
                doc_parts.append(f"SYMPTOMS: {', '.join(symptoms)}")
            if medicines:
                doc_parts.append(f"MEDICINES: {', '.join(medicines)}")
            if precautions:
                doc_parts.append(f"PRECAUTIONS: {', '.join(precautions)}")
            
            comprehensive_doc = ". ".join(doc_parts) + "."
            doc_variations.append(comprehensive_doc)
            
            # Variation 2: Symptoms-focused
            if symptoms:
                symptoms_doc = f"{disease} SYMPTOMS: {', '.join(symptoms)}. Common signs include {', '.join(symptoms[:3])}."
                doc_variations.append(symptoms_doc)
            
            # Variation 3: Treatment-focused
            if medicines:
                treatment_doc = f"{disease} TREATMENT: Medications include {', '.join(medicines[:3])}. {', '.join(medicines[:2])} are commonly prescribed."
                doc_variations.append(treatment_doc)
            
            # Variation 4: Prevention-focused
            if precautions:
                prevention_doc = f"{disease} PREVENTION: {', '.join(precautions[:3])}. Recommended precautions: {', '.join(precautions[:2])}."
                doc_variations.append(prevention_doc)
            
            # Add all variations to documents
            for doc in doc_variations:
                documents.append(doc)
                metadata.append({
                    'disease': disease,
                    'symptoms': symptoms,
                    'medicines': medicines,
                    'precautions': precautions
                })
            
            # Show first few entries
            if len(documents) <= 6:  # Show first 2 diseases (3 variations each)
                print(f"   📝 Disease: {disease}")
                for i, doc in enumerate(doc_variations[:2], 1):
                    print(f"      Doc {i}: {doc[:100]}...")
        
        print(f"\n✅ Created {len(documents)} knowledge documents")
        print(f"✅ Created {len(metadata)} metadata entries")
        
        return documents, metadata
    
    def create_faiss_index(self):
        """Create and save FAISS vector index"""
        print("="*60)
        print("🏥 HEALTHCARE FAISS INDEX CREATION")
        print("="*60)
        
        # Create knowledge base
        documents, metadata = self.create_knowledge_base()
        
        if not documents:
            print("\n❌ CRITICAL: No documents created!")
            print("Please check your health.json file format.")
            print("Expected format:")
            print('{"diseases": [{"name": "Flu", "symptoms": ["Fever", "cough"], "medicines": ["Paracetamol"], "precautions": ["Rest"]}, ...]}')
            return False
        
        # Load embedding model
        print("\n⚡ Loading embedding model...")
        try:
            embedder = SentenceTransformer(self.embed_model)
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return False
        
        # Generate embeddings
        print("⚡ Generating embeddings...")
        try:
            embeddings = embedder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            print(f"✅ Generated {embeddings.shape[0]} embeddings")
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return False
        
        # Create FAISS index
        print("🏗️ Creating FAISS index...")
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype(np.float32))
            print(f"✅ FAISS index created with {index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False
        
        # Save index and metadata
        print("💾 Saving FAISS index...")
        try:
            os.makedirs(self.faiss_folder, exist_ok=True)
            
            faiss.write_index(index, os.path.join(self.faiss_folder, "index.faiss"))
            with open(os.path.join(self.faiss_folder, "docs.pkl"), "wb") as f:
                pickle.dump(documents, f)
            with open(os.path.join(self.faiss_folder, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            print(f"✅ All files saved in: {self.faiss_folder}/")
            print("🎉 FAISS index creation completed successfully!")
            print(f"📊 Total diseases: {len(set(m['disease'] for m in metadata))}")
            print(f"📚 Total documents: {len(documents)}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving files: {e}")
            return False

if __name__ == "__main__":
    creator = FAISSCreator()
    success = creator.create_faiss_index()
    
    if not success:
        print("\n❌ Failed to create FAISS index!")
        print("\n📋 TROUBLESHOOTING:")
        print("1. Ensure health.json is in data/ folder")
        print("2. Check JSON format is valid")
        print("3. Verify structure: {'diseases': [{'name': '...', 'symptoms': [], ...}, ...]}")
        print("4. Make sure file is not empty")
        exit(1)