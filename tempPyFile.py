import mechanize
from bs4 import BeautifulSoup
from html2text import html2text
import re
from nltk.tokenize import sent_tokenize
import sqlite3


def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\r+", " ", cleaned)
    cleaned = re.sub(r"\t+", " ", cleaned)
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)

    return cleaned.strip()


def download_text_from_web(url):
    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]
    html = br.open(url).read().decode('utf-8')
    soup = BeautifulSoup(html,features="html5lib")
    cleanhtml = clean_html(html)
    return sent_tokenize(cleanhtml)

conn = sqlite3.connect('new_DBfile')
c = conn.cursor()

def download_links_from_DB():
    urllist = []
    for row in c.execute('select * from PPs_links'):  # reads the data at once
        urllist.append(row[0])
    return urllist

def stor_in_db(data):
    c.execute("""insert into PPs_Sensitive values ('%s')""" %data)
    conn.commit()
