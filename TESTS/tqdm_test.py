from tqdm import tqdm
from time import sleep

def process(file):
    sleep(0.1)

files = [i for i in range(100)]

for file in tqdm(files):
    process(file)