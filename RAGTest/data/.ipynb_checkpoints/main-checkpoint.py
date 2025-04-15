import json

documents = []
with open("/home/LAB/maoqr/hzt_RAG/zh/ragx/data/documents.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
for entry in data:
    
    title = entry['title']
    print(title)
    
    if not title.endswith('法'):
        print(f"Error: 标题 '{title}' 不是以'法'结尾")
    
    text = entry['title'] + entry['text']
    id = entry['id']
    
