import json

if __name__ == "__main__":
    files = ["train", "test", "dev"]
    ch_path = "transformer-pytorch/data/corpus.ch"
    en_path = "transformer-pytorch/data/corpus.en"
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open("transformer-pytorch/data/dataset/" + file + ".json", "r", encoding="utf-8"))  # noqa: SIM115
        for item in corpus:
            en_lines.append(item[0] + "\n")
            ch_lines.append(item[1] + "\n")

    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)

    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)



    print("lines of chinese: ",len(ch_lines))
    print("lines of english: ",len(en_lines))

    
