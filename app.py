# Simple FAQ Chatbot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQs (questions + answers)
questions = [
    "what is python",
    "how are you",
    "what is your name",
    "what is machine learning",
    "what is sql",
    "what is cloud computing",
    "what is operating system"
]

answers = [
    "Python is a programming language",
    "I am fine",
    "I am a chatbot"
    "Machine learning is a part of AI that learns from data" 
    "SQL is used to manage databases",
    "Cloud computing means storing data online",
    "Operating system controls computer"
]

# Convert questions into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chatbot function
def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

# Run chatbot
while True:
    user = input("You: ")
    if user == "exit":
        print("Bot: Bye!")
        break
    print("Bot:", chatbot(user))