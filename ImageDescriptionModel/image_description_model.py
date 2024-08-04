import base64
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class ImageDescriptionModel():
    def __init__(self, openai_api_key):
        self.chat_model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_prompt(self, param_dict):
        system_message = """
            You are an advanced AI assistant tasked with analyzing images.
            The answer to the question:
            - should start with a general overview of the image,
            - should provide specific information about the objects and people in the image.
            Sample response: 'The image shows a person holding a knife with a blade that has text engraved on it. The person is wearing a light-colored dress with lace details on the sleeves. Their hair is long and blonde.'
        """
        human_messages = [
            {"type": "text", "text": f"{param_dict['question']}"},
            {"type": "image_url", "image_url": {"url" : f"{param_dict['image_url']}"}}
        ]
        return [SystemMessage(content=system_message), HumanMessage(content=human_messages)]

    def get_image_description_and_object(self, image_path):
        base64_image = self.encode_image(image_path)
        chain = self.generate_prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"question": "What is depicted in this image?", "image_url": f"data:image/jpeg;base64, {base64_image}"})
        return  result
