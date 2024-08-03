import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def chat_with_mistral(user_input):
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-large-latest"  # Use "Mistral-7B-v0.2" for "mistral-tiny"

    client = MistralClient(api_key=api_key)
    messages = [ChatMessage(role="user", content=user_input)]

    chat_response = client.chat(model=model, messages=messages)
    return chat_response.choices[0].message.content


iface = gr.Interface(
    fn=chat_with_mistral,
    inputs=gr.components.Textbox(label="Enter Your Message"),
    outputs=gr.components.Markdown(label="Chatbot Response"),
    title="Mistral AI Chatbot",
    description="Interact with the Mistral API via this chatbot. Enter a message and get a response.",
    examples=[["Give me a meal plan for today"]],
    allow_flagging="never",
)

iface.launch()
