import os

dir_early = "./_data/out"
dir_later = "./out"
output_directory = "./_data/concat"

files = set(os.listdir(dir_early)) | set(os.listdir(dir_later))

for file_name in files:
    print(f"- combining {file_name}")
    file_early = os.path.join(dir_early, file_name)
    file_later = os.path.join(dir_later, file_name)

    content = ""

    # Try adding early file.
    try:
        with open(file_early, "r") as file:
            content += file.read()
    except Exception as err:
        print(err)

    # Try adding later file.
    try:
        with open(file_later, "r") as file:
            content += file.read()
    except Exception as err:
        print(err)

    output_path = os.path.join(output_directory, file_name)
    with open(output_path, "w") as output_file:
        output_file.write(content)
