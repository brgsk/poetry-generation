from collections import defaultdict
from pathlib import Path

import hydra
import jsonlines
import requests
from bs4 import BeautifulSoup
from fire import Fire
from logzero import logger
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track

console = Console(force_terminal=True)


def mine(url: str, author: str, filepath="res/data/poems") -> None:
    """Finds all poems written by the author and saves them to a file.

    :param url: author's page on `poezja.org`
    :param author: author's name used as file name.
    :param filepath: path to directory where scrapped texts will be saved to
    """
    logger.info(f"Sending requests")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Could not connect\nStatus_code: {response.status_code}")
    else:
        logger.debug(f"Status code: {response.status_code}")

        logger.info(f"Downloading from {url} to {filepath}/{author}.jsonl")
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        logger.debug("Parsing links to poems")
        links = get_links(soup, url)
        poems_path = Path(filepath)
        if not poems_path.exists():
            poems_path.mkdir(parents=True)
        logger.debug("Loading banned poems")
        banned = _get_banned()
        if author in banned.keys():
            banned_poems = banned[author]
        with jsonlines.open((poems_path / f"{author}.jsonl"), "w") as file:
            for link in track(links, description="Processing texts..."):
                if "#" not in link:
                    try:
                        page = requests.get(link).content
                        poem_soup = BeautifulSoup(page, "html.parser")
                        title, poem = get_poem(poem_soup)
                        if author in banned.keys():
                            if title in banned_poems:
                                continue
                        if (
                            "english" not in title
                            and "esperanto" not in title
                            and "[en]" not in title
                        ):
                            file.write({"title": title, "text": poem})
                    except requests.exceptions.ConnectionError:
                        print("Failed to connect with %s." % link)


def _get_banned(filename="restricted.jsonl") -> dict:
    if Path(filename).exists():
        with jsonlines.open(filename) as restricted:
            banned_poems = defaultdict(list)
            for line in restricted:
                banned_poems[line["author"]].append(line["title"])
        return banned_poems


def _get_poems_for_kids(url, url2, author, filepath="res/data/poems") -> None:
    """Finds all poems written by the author and saves them to a file.

    :param url: Author's page
    :param author: Author's name
    :param filepath: Name of a generated file
    """
    logger.info(f"Downloading from {url} to {filepath}/restricted.jsonl")
    logger.info(f"Sending requests")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Could not connect\nStatus_code: {response.status_code}")
    else:
        logger.info("ok")
        logger.debug(f"Status code: {response.status_code}")
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        logger.debug("Parsing links to poems")
        links = get_links(soup, url2)
        poems_path = Path(filepath)
        if not poems_path.exists():
            poems_path.mkdir(parents=True)
        with jsonlines.open((poems_path / "restricted.jsonl"), "a") as file:
            for link in track(links, description="Processing texts..."):
                if "#" not in link:
                    try:
                        page = requests.get(link).content
                        poem_soup = BeautifulSoup(page, "html.parser")
                        title, poem = get_poem(poem_soup)
                        if (
                            "english" not in title
                            and "esperanto" not in title
                            and "[en]" not in title
                        ):
                            file.write({"author": author, "title": title})
                    except requests.exceptions.ConnectionError:
                        print("Failed to connect with %s." % link)


def get_links(soup, url) -> list[str]:
    """Parses a page to find links to poems.

    :param soup: BeautifulSoup object with the parsed HTML
    :param url: Author's page
    :return: List of urls with poems
    """
    return [
        a.get("href")
        for ul in soup.find_all("ul")
        for a in ul.find_all("a")
        if a.get("href").startswith(url)
    ]


def get_poem(soup) -> tuple[str, str]:
    """Parses a page to get the title and the content of a poem.

    :param soup: BeautifulSoup object with the parsed HTML
    :return: 2-tuple containing the title and the content of the poem.
    """
    title = soup.title.text
    title = title.split(" - ")[0]
    poem = soup.find(attrs={"class": "row justify-content-center lyric-entry"}).text
    poem_end = poem.rfind("Czytaj dalej:")
    poem = poem[:poem_end]
    return title, poem


def fire_mine() -> None:
    Fire(mine)


if __name__ == "__main__":
    fire_mine()
