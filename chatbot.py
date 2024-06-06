import streamlit as st
import boto3
import json


# Create a function to send messages to the chatbot model
def send_message(user_input):
    # Initialize the Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Specify the model ID
    model_id = "amazon.titan-text-premier-v1:0"

    # Define the prompt for the model, adjusted for chat interactions
    prompt = "The following conversation is with a chatbot. Please respond appropriately:\\n\\n" \
             "Human: " + user_input + "\\n" \
                                      "Bot:"

    # Prepare the request payload
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 150,  # Adjust max tokens for chatbot responses
            "temperature": 0.7,  # Slightly higher temperature for varied responses
        },
    }

    # Convert the native request to JSON
    request = json.dumps(native_request)

    # Invoke the model with the request
    response = client.invoke_model(modelId=model_id, body=request)

    # Decode the response body
    model_response = json.loads(response["body"].read())

    # Extract the chatbot response
    response_text = model_response["results"][0]["outputText"]
    return response_text


# Streamlit user interface
st.title("Chatbot Interface")
user_input = st.text_input("Talk to the chatbot:")

if user_input:
    # Get the response from the model
    model_response = send_message(user_input)
    st.text_area("Chatbot says:", value=model_response, height=200)
