import streamlit as st
import requests

BACKEND_URL = "http://localhost:9999"

def upload_pdf(uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post(f"{BACKEND_URL}/upload-pdf/", files=files)
    return response.json()

def send_chat_message(message):
    payload = {"message": message}
    response = requests.post(f"{BACKEND_URL}/chat/", json=payload)
    return response.json()

def main():
    st.title("Upload a PDF and ask questions!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Upload PDF"):
        try:
            response = upload_pdf(uploaded_file)
            st.success(f"Uploaded successfully! Text length: {response['text_length']}")
            st.session_state.pdf_uploaded = True
        except Exception as e:
            st.error(f"Upload failed: {e}")

    # Display messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        if not st.session_state.pdf_uploaded:
            st.error("Please upload a PDF first!")
        else:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            try:
                response = send_chat_message(prompt)
                result = response.get("response", "No answer found.")
                
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()