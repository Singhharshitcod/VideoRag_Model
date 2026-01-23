# Converts the videos mp4 to mp3 
import os 
import subprocess

files = os.listdir("RAG_MODEL/videosml") 
for file in files: 
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" | ")[0]
    print( tutorial_number,  file_name)
    subprocess.run(["ffmpeg", "-i", f"RAG_MODEL/videosml/{file}", f"RAG_MODEL/audiosml/{tutorial_number}_{file_name}.mp3"])
print("Videos Converted to Audio Sucessfully !")    