# ADBOT
## Overview
This repository belongs to the "аддикция" community on VK (https://vk.com/addiktsiya)
## Usage
### Preparation
Before using the repository tools, you must create a weights folder inside the SAM folder and execute the following code inside that folder:
```
wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
```
You also need to create a database, an extension, and a table:
```
CREATE DATABASE YOUR_DATABASE_NAME;
CREATE EXTENSION vector;
CREATE TABLE YOUR_TABLE_NAME (id bigserial PRIMARY KEY, name text, embedding vector(768), song_fragment text);
```
### Example of usage
Adding a song to the database:
```
python main.py --insert "lil peep - save that sheet"
```
Generation a text for control image
```
python main.py --show
```
Generation an image
```
python main.py --object "a can of soda" --number 4
```
