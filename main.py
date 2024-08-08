import argparse
from PIL import Image
from DINO.dino import DINO
from SAM.sam import SAM
from ImageDescriptionModel.image_description_model import ImageDescriptionModel
from Database.database import Database
from ImageGenerationModel.image_generation_model import ImageGenerationModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import cv2
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--insert", type=str, help="Add a song to the database")
    parser.add_argument("--show", action="store_true", help="Show the text that is needed to create a control image")
    parser.add_argument("--object", type=str, help="The object to be redrawn")
    parser.add_argument("--number", type=int, help="The number of images to be generated")
    parser.add_argument("--strength", type=float, help="The strength of image to image generation")
    args = parser.parse_args()

    load_dotenv()

    if args.insert is not None:
        # Getting the values of environment variables
        database_name = os.getenv("DATABASE_NAME")
        table_name = os.getenv("TABLE_NAME")
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")
        # Connecting to database
        database = Database(
            database_name,
            table_name,
            user,
            password,
            host
        )
        # Initializing embeddings generation model
        hf = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-base-en-v1.5",
            model_kwargs = {"device":'cpu'},
            encode_kwargs = {'normalize_embeddings':True}
        )
        song_name = args.insert
        def len_func(text):
            return len(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=50,
            length_function=len_func,
            is_separator_regex= False
        )
        with open(f"Songs/{song_name}.txt", 'r') as file:
            song = file.read().strip()
        song_fragments = text_splitter.split_text(song)
        embeddings = [np.array(hf.embed_query(song_fragment)) for song_fragment in song_fragments]
        data = [(song_name, embedding, song_fragment) for embedding, song_fragment in zip(embeddings, song_fragments)]
        database.insert(data)
        database.disconnect()

    if args.show:
        # Getting the values of environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        database_name = os.getenv("DATABASE_NAME")
        table_name = os.getenv("TABLE_NAME")
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")
        # Connecting to database
        database = Database(
            database_name,
            table_name,
            user,
            password,
            host
        )
        # Initializing embeddings generation model
        hf = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-base-en-v1.5",
            model_kwargs = {"device":'cpu'},
            encode_kwargs = {'normalize_embeddings':True}
        )
        image_path = "Content/image.png"
        image_description_model = ImageDescriptionModel(openai_api_key)
        image_description = image_description_model.get_image_description_and_object(image_path)
        embedding = np.array(hf.embed_query(image_description))
        database.similarity_search(embedding)
        database.disconnect()

    if args.object is not None and args.number is not None and args.strength is not None:
        object = args.object
        prompt = args.object
        number = args.number
        image_for_dino = Image.open("Content/image.png")
        image_for_image_generation_model = Image.open("Content/image.png")
        image_for_sam = cv2.cvtColor(cv2.imread("Content/image.png"), cv2.COLOR_BGR2RGB)
        control = Image.open("Content/control.png")
        dino = DINO()
        box = dino.get_box(image_for_dino, object)
        sam = SAM()
        mask = sam.get_mask(image_for_sam, box)
        image_generation_model = ImageGenerationModel()
        for i in range(number):
            image_generation_model.generate(prompt, image_for_image_generation_model, mask, control, i, args.strength)

if __name__ == "__main__":
    main()
