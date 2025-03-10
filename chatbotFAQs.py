import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# Vérifier si un GPU est disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger BERT et le tokenizer une seule fois
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)  # Charger sur GPU si disponible


# Base de connaissances FAQ
faq_data = {
    "What are your business hours?": "Our business hours are from 9 AM to 6 PM, Monday to Friday.",
    "How can I contact customer support?": "You can contact our support team at support@example.com or call +1-234-567-890.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and bank transfers.",
    "How can I track my order?": "You can track your order by logging into your account and visiting the 'Orders' section."   
}


# Fonction pour obtenir un embedding BERT (moyenne des tokens)
def get_embedding(text):
    inputs = tokenizer(text.lower(), return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}  # Déplacer vers GPU si disponible
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1)  # Moyenne des tokens pour obtenir un vecteur global


# Calculer et stocker les embeddings des questions FAQ
faq_embeddings = {q: get_embedding(q) for q in faq_data.keys()}


# Trouver la meilleure réponse en fonction de la similarité cosinus
def get_best_response(user_input):
    try:
        user_embedding = get_embedding(user_input)

        best_match = None
        best_score = -1

        for question, embedding in faq_embeddings.items():
            score = F.cosine_similarity(user_embedding, embedding).item()
            if score > best_score:
                best_score = score
                best_match = question

        return faq_data.get(best_match, "Sorry, I couldn't find an answer.")
    
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Fonction principale du chatbot
def chatbot():
    print("Hello! I am your FAQ chatbot. Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        if not user_input:
            print("Chatbot: Please enter a valid question.")
            continue
        
        response = get_best_response(user_input)
        print(f"Chatbot: {response}")

if __name__=="__main__":
    chatbot()


