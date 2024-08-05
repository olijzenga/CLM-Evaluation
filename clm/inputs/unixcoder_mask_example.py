def <mask>(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
