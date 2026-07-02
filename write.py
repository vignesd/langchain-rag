

def write_data_to_file(data_type,data, file_path):
    """
    Write data to a file.

    Args:
        data (str): The data to write to the file.
        file_path (str): The path to the file where the data will be written.
    """
    with open(file_path, 'a',encoding='utf-8') as f:
        f.write(f"{data_type}: {data}\n")  # Write the data type and data to the file
        f.write("-----"*20 + "\n")  # Add an extra newline for separation