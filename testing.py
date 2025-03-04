import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from huggingface_hub import login  # For authentication

# Load the DeepSeek-R1 model and tokenizer
def load_model():
    print("Loading DeepSeek-R1 model...")
    
    # Authenticate with Hugging Face (if required)
    # Replace "your_huggingface_token" with your actual token
    login(token="your_huggingface_token")
    
    # Load the model and tokenizer
    MODEL_NAME = "deepseek-ai/DeepSeek-R1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")
    return tokenizer, model, device

# Generate a response using the DeepSeek-R1 model
def generate_response(prompt, tokenizer, model, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# CLI interface for the chatbot
def cli_chatbot(tokenizer, model, device):
    print("Welcome to the DeepSeek-R1 Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(user_input, tokenizer, model, device)
        print(f"DeepSeek-R1: {response}")

# Flask web app for the chatbot
def create_flask_app(tokenizer, model, device):
    app = Flask(__name__)

    @app.route("/chat", methods=["POST"])
    def chat():
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        response = generate_response(user_input, tokenizer, model, device)
        return jsonify({"response": response})

    return app

# Main function to run the app
def main():
    # Load the model
    tokenizer, model, device = load_model()

    # Choose between CLI or Flask app
    print("Choose an option:")
    print("1. Command Line Interface (CLI)")
    print("2. Web App (Flask)")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        cli_chatbot(tokenizer, model, device)
    elif choice == "2":
        app = create_flask_app(tokenizer, model, device)
        print("Starting Flask app. Access the API at http://127.0.0.1:5000/chat")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()