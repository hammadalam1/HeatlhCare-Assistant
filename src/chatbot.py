# chatbot.py
import streamlit as st
from diagnosis import HealthcareDiagnoser
import time

class HealthcareChatbot:
    def __init__(self):
        st.set_page_config(
            page_title="Healthcare Diagnosis Chatbot",
            page_icon="🏥",
            layout="wide"
        )
        
        # Initialize diagnoser
        try:
            self.diagnoser = HealthcareDiagnoser()
            st.session_state.diagnoser_loaded = True
        except Exception as e:
            st.error(f"Error loading diagnosis system: {e}")
            st.session_state.diagnoser_loaded = False
    
    def display_diagnosis(self, diagnosis_result):
        """Display complete diagnosis results with all information"""
        diagnoses = diagnosis_result['diagnoses']
        
        if not diagnoses:
            st.warning(diagnosis_result['message'])
            return
        
        st.success(diagnosis_result['message'])
        
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("🤒 Diagnosis & Symptoms")
            for disease in diagnoses:
                with st.expander(f"**{disease['disease']}** ({disease['confidence']:.1f}% confidence)"):
                    st.write("**Symptoms:**")
                    for symptom in disease['symptoms']:
                        st.write(f"• {symptom}")
        
        with col2:
            st.subheader("💊 Recommended Medicines")
            for disease in diagnoses:
                with st.expander(f"**{disease['disease']}**"):
                    if disease['medicines']:
                        st.write("**Medicines:**")
                        for medicine in disease['medicines']:
                            st.write(f"• {medicine}")
                    else:
                        st.info("No specific medicines listed")
        
        with col3:
            st.subheader("🛡️ Precautions & Prevention")
            for disease in diagnoses:
                with st.expander(f"**{disease['disease']}**"):
                    if disease['precautions']:
                        st.write("**Precautions:**")
                        for precaution in disease['precautions']:
                            st.write(f"• {precaution}")
                    else:
                        st.info("No specific precautions listed")
    
    def display_compact_view(self, diagnosis_result):
        """Alternative compact view"""
        diagnoses = diagnosis_result['diagnoses']
        
        for i, disease in enumerate(diagnoses, 1):
            st.subheader(f"{i}. {disease['disease']} ({disease['confidence']:.1f}% confidence)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Symptoms:**")
                for symptom in disease['symptoms'][:5]:  # Show first 5 symptoms
                    st.write(f"• {symptom}")
            
            with col2:
                st.write("**Medicines:**")
                if disease['medicines']:
                    for medicine in disease['medicines'][:3]:  # Show first 3 medicines
                        st.write(f"• {medicine}")
                else:
                    st.info("No medicines listed")
            
            with col3:
                st.write("**Precautions:**")
                if disease['precautions']:
                    for precaution in disease['precautions'][:3]:  # Show first 3 precautions
                        st.write(f"• {precaution}")
                else:
                    st.info("No precautions listed")
            
            st.markdown("---")
    
    def run(self):
        """Run the Streamlit application"""
        st.title("🏥 Healthcare Diagnosis Chatbot")
        st.markdown("Enter your symptoms below to get AI-powered diagnosis with complete medicine and precaution information.")
        
        # Initialize session state
        if 'diagnosis_history' not in st.session_state:
            st.session_state.diagnosis_history = []
        
        # Symptoms input
        symptoms = st.text_area(
            "**Describe your symptoms:**",
            placeholder="e.g., fever, cough, headache, fatigue, body ache...",
            height=100,
            help="Be as specific as possible about your symptoms"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 Diagnose", type="primary", use_container_width=True) and symptoms:
                if not st.session_state.diagnoser_loaded:
                    st.error("Diagnosis system not loaded. Please check if FAISS index exists.")
                    return
                
                with st.spinner("Analyzing symptoms and retrieving medical information..."):
                    time.sleep(1)  # Simulate processing
                    diagnosis_result = self.diagnoser.generate_response(symptoms)
                
                # Save to history
                st.session_state.diagnosis_history.append({
                    'symptoms': symptoms,
                    'result': diagnosis_result,
                    'timestamp': time.time()
                })
                
                st.session_state.current_diagnosis = diagnosis_result
        
        with col2:
            if st.button("🧹 Clear", use_container_width=True):
                if 'current_diagnosis' in st.session_state:
                    del st.session_state.current_diagnosis
                st.rerun()
        
        # Display current diagnosis
        if 'current_diagnosis' in st.session_state:
            st.markdown("---")
            st.subheader("📋 Diagnosis Results")
            
            # Toggle between views
            view_option = st.radio(
                "View Mode:",
                ["Detailed View", "Compact View"],
                horizontal=True
            )
            
            if view_option == "Detailed View":
                self.display_diagnosis(st.session_state.current_diagnosis)
            else:
                self.display_compact_view(st.session_state.current_diagnosis)
            
            # Disclaimer
            st.warning("""
            ⚠️ **Important Medical Disclaimer:**  
            This is an AI-assisted diagnosis tool and should **NOT** replace professional medical advice.  
            The information provided is based on pattern matching and may not be accurate for your specific case.  
            **Always consult a qualified healthcare provider** for proper diagnosis and treatment.
            """)
        
        # Diagnosis history sidebar
        with st.sidebar:
            st.header("📋 Diagnosis History")
            
            if st.session_state.diagnosis_history:
                for i, history in enumerate(reversed(st.session_state.diagnosis_history[-5:]), 1):
                    with st.expander(f"Diagnosis {i}: {history['symptoms'][:30]}..."):
                        st.write(f"**Symptoms:** {history['symptoms']}")
                        if history['result']['diagnoses']:
                            st.write("**Possible Conditions:**")
                            for disease in history['result']['diagnoses']:
                                st.write(f"- {disease['disease']} ({disease['confidence']:.1f}%)")
            
            st.header("ℹ️ About")
            st.info("""
            **How it works:**
            - Uses RAG (Retrieval Augmented Generation) with FAISS vector database
            - Diagnoses diseases based on your symptoms
            - Retrieves **complete medical information** including:
              - All symptoms for each disease
              - All recommended medicines  
              - All precautions and prevention tips
            - Provides confidence scores for each diagnosis
            """)
            
            st.header("⚙️ Settings")
            if st.button("🔄 Rebuild FAISS Index"):
                try:
                    from create_faiss import FAISSCreator
                    creator = FAISSCreator()
                    if creator.create_faiss_index():
                        st.success("FAISS index rebuilt successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding index: {e}")

if __name__ == "__main__":
    chatbot = HealthcareChatbot()
    chatbot.run()