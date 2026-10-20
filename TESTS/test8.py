try:
    with open("misssing.txt") as f:
        print(f.read())
except OSError as error:
    print(error)