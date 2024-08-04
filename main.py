import argparse
from PIL import Image
from DINO.dino import DINO
from SAM.sam import SAM
from ImageDescriptionModel.image_description_model import ImageDescriptionModel
from Database.database import Database
from ImageGenerationModel.image_generation_model import ImageGenerationModel
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import cv2
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--insert", type=str, help="")
    parser.add_argument("--show", action="store_true", help="")
    parser.add_argument("--object", type=str, help="")
    parser.add_argument("--number", type=str, help="")
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
        with open("Content/song.txt", 'r') as file:
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
        print("Getting an image description using ImageDescriptionModel ⏳ ", end='')
        image_description_model = ImageDescriptionModel(openai_api_key)
        image_description = image_description_model.get_image_description_and_object(image_path)
        print("\b✅ ")
        embedding = np.array(hf.embed_query(image_description))
        song_fragment = database.similarity_search(embedding)
        print(song_fragment)
        database.disconnect()

    if args.object is not None and args.number is not None:
        object = args.object
        prompt = args.object
        number = args.number
        image_for_dino = Image.open("Content/image.png")
        image_for_image_generation_model = Image.open("Content/image.png")
        image_for_sam = cv2.cvtColor(cv2.imread("Content/image.png"), cv2.COLOR_BGR2RGB)
        control = Image.open("Content/control.png")
        print("Getting a box using GroundingDINO ⏳ ", end='')
        dino = DINO()
        box = dino.get_box(image_for_dino, object)
        print("\b✅ ")
        print("Getting a mask using SAM ⏳ ", end='')
        sam = SAM()
        mask = sam.get_mask(image_for_sam, box)
        print("\b✅ ")
        print("Generating ⏳", end='')
        image_generation_model = ImageGenerationModel()
        for i in range(number):
            image_generation_model.generate(prompt, image_for_image_generation_model, mask, control, i)
            print("\b✅ ")

if __name__ == "__main__":
    main()
