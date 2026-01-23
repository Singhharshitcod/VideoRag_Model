import os
import math
import json

n=5 #size of chunks grouping

for filename in os.listdir("RAG_MODEL/jsons"):
  if filename.endswith(".json"):
     filepath=os.path.join("RAG_MODEL/jsons",filename)
     with open(filepath,"r", encoding="utf-8") as f:
          data=json.load(f)
          new_chunks=[] # store grouped chunks for one json file. ie, 1 mergejson file ka content 
          num_chunks=len(data["chunks"])
          num_group=math.ceil(num_chunks/n)# 2.5=3

          for i in range(num_group):
              start_idx=i*n
              end_idx=min((i+1)*n,num_chunks)

              grouped_chunk=data["chunks"][start_idx:end_idx]

              new_chunks.append({
                  "number":data["chunks"][0]["number"], #video number ,not chunk number
                  "title":data["chunks"][0]["title"],
                  "start":grouped_chunk[0]["start"],
                  "end":grouped_chunk[-1]["end"],
                  "text":" ".join(c["text"] for c in grouped_chunk)
            })

          mergefilepath=os.path.join("RAG_MODEL/mergejsons",filename)

          with open(mergefilepath,"w") as f:
              json.dump({
                  "chunks":new_chunks,
                  "text":data["text"]
              },f,indent=4)

print("All Chunks Merged Sucessfully !")

            

