import re
from pathlib import Path

import jsonlines
import pandas as pd
from fire import Fire


def process_poems(data_path: str) -> None:
    data_path = Path(data_path)
    poems = _loadpoems(data_path)
    texts = [poem["text"].strip() for poem in poems]
    titles = [poem["title"] for poem in poems]

    df = pd.DataFrame(list(zip(titles, texts)), columns=["title", "text"])
    stanza_title, stanza_text = _split_into_stanzas(df["title"].values, df["text"].values)
    stanza_df = pd.DataFrame(list(zip(stanza_title, stanza_text)), columns=["title", "stanza_text"])

    line_title, line_text = _split_into_lines(
        stanza_df["title"].values, stanza_df["stanza_text"].values
    )
    line_df = pd.DataFrame(list(zip(line_title, line_text)), columns=["title", "line_text"])

    line_df.to_csv(data_path / "lines.csv", index=False)
    stanza_df.to_csv(data_path / "stanzas.csv", index=False)


def _loadpoem(filename: Path) -> list:
    filename = str(filename)
    poems = []
    if ".jsonl" in filename and filename != "restricted.jsonl":
        with jsonlines.open(filename) as reader:
            for poem in reader:
                poems.append(poem)
    return poems


def _loadpoems(path: Path):
    all_poems = []
    for file in path.iterdir():
        poems = _loadpoem(file)
        all_poems.extend(poems)
    return all_poems


def _clean_poem_text(poem_text: str) -> list:
    poem_text = poem_text.strip()
    poem_text = re.sub(" +", " ", poem_text)
    poem_text = re.sub("\n\n\r\n\r\n", "\n\r\n \n\r\n", poem_text)
    poem_text = re.sub("\n\r\n", "\n", poem_text)
    poem_text = re.sub(" +", " ", poem_text)
    return poem_text


def _split_into_stanzas(poem_title_list: list, poem_text_list: list) -> tuple[list, list]:
    return_title_list = []
    return_stanza_list = []
    for poem_index in range(len(poem_title_list)):
        poem_text = _clean_poem_text(poem_text_list[poem_index])
        poem_stanzas = poem_text.split("\n\n")
        for stanza in poem_stanzas:
            return_title_list.append(poem_title_list[poem_index])
            return_stanza_list.append(stanza)
    return return_title_list, return_stanza_list


def _split_into_lines(poem_title_list: list, poem_text_list: list) -> tuple[list, list]:
    return_title_list = []
    return_line_list = []
    for poem_index in range(len(poem_title_list)):
        poem_lines = poem_text_list[poem_index].split("\n")
        for line in poem_lines:
            return_title_list.append(poem_title_list[poem_index])
            return_line_list.append(line.strip())
    return return_title_list, return_line_list


def fire_process() -> None:
    Fire(process_poems)


if __name__ == "__main__":
    fire_process()
