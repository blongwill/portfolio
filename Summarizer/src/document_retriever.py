#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"

from lxml import etree
from lxml import html
from lxml.etree import XMLParser, parse

'''#####################################
-Class Object used to more easily retrieve raw document data from Acquaint and Acquaint2 Databases.
-Configures document ID to create document file path and to determine database
-Determines HTML or XML parsing and Tags used for internal element searching
-Caches XML/HTML files as they are parsed in Dictionary with path-file format 
'''#######################################

class Document_Retriever:
    def __init__(self):
        self.xml_parser_cache = {}
        self.date=None
        self.headline_tag = None
        self.category_tag = None
        self.dateline_tag = None
        self.text_tag = None

    #Contains lots of the hardcoded configurations in order to parse doc id to derive the correct database
    def configure(self, doc_id: str):
        self.doc_id = doc_id

        if "_" in doc_id:
            source, lang, other = doc_id.split("_")
            self.date, specifier = other.split(".")
            year = self.date[:4]
            month = self.date[4:6]
        else:
            source = doc_id[:3]
            self.date = doc_id[3:11]
            year = self.date[:4]
            month= self.date[4:6]
            # specifier = doc_id[-4:]
            lang = "ENG"

        if source == "XIE":
            alt_source = "XIN"
        else:
            alt_source = source

        #############################
        self.acquaint = int(year) >= 1996 and int(year) <= 2000

        # The year 2006 is represented in both databases. October appears to be the month where the division is made
        self.acquaint2 = int(year) >= 2004 and int(year) < 2006 or (int(year) == 2006 and int(month) < 10)

        self.gigaword = int(year) >= 2007 and int(year) <= 2008 or (int(year)==2006 and int(month)>=10)

        if self.acquaint:

            if alt_source != 'NYT':
                alt_source = alt_source + "_ENG"

            self.headline_tag = 'headline'
            self.category_tag = 'category'
            self.dateline_tag = 'date_time'
            self.text_tag = 'text'

            self.doc_path = "/corpora/LDC/LDC02T31/" + source.lower() + "/" + year + "/" + self.date + "_" + alt_source.upper()

        elif self.acquaint2 or self.gigaword:

            self.headline_tag = 'HEADLINE'
            self.dateline_tag = 'DATELINE'
            self.text_tag = 'TEXT'
            self.category_tag = None

            if self.acquaint2:
                self.doc_path = "/corpora/LDC/LDC08T25/data/" + alt_source.lower() + "_" + lang.lower() + "/" + alt_source.lower() + "_" + lang.lower() + "_" + self.date[:6] + ".xml"
            else:
                self.doc_path = "/corpora/LDC/LDC11T07/data/" + alt_source.lower() + "_" + lang.lower() + "/" + alt_source.lower() + "_" + lang.lower() + "_" + self.date[:6] + ".gz"

    #Method requires document ID to determine database and document file path.
    # Retrieves raw document.
    def retrieve_doc(self, doc_id):
        self.configure(doc_id)
        raw_doc = None
        tree = self.xml_parser_cache.get(self.doc_path)

        if self.acquaint:

            if tree is None:
                parser = etree.HTMLParser(encoding='utf-8', remove_blank_text=True)
                with open(self.doc_path) as file:
                    tree = html.fragment_fromstring(file.read(), create_parent='body', parser=parser)

            raw_doc = [element for element in tree.findall("doc") if element.find("docno").text == " " + doc_id + " "][0]
            raw_doc.getparent().remove(raw_doc)  # Removes previous accessed raw document from tree to save memory in cache

        elif self.acquaint2:

            if tree is None:
                tree = etree.parse(self.doc_path)

            raw_doc = [element for element in tree.findall("DOC") if element.get("id") == doc_id][0]
            #### Must not remove document from tree because documents repeat under different topics

        elif self.gigaword:

            if tree is None:
                p = XMLParser(huge_tree=True) #### Some files are too large, without this they prevent parsing

                with gzip.open(self.doc_path, 'rt', encoding='latin-1') as file:
                    data=file.read()

                    #### In this one file there's a less than symbold that prevents parsing
                    if self.doc_path =='/corpora/LDC/LDC11T07/data/xin_eng/xin_eng_200811.gz':
                        data = data.replace('<3', 'lt 3') ### Replaces the < with lt

                    tree = etree.fromstring('<DOCSTREAM>\n' + data.strip() + '\n</DOCSTREAM>\n',parser=p)


            raw_doc = [element for element in tree.findall("DOC") if element.get("id") == doc_id][0]
            raw_doc.getparent().remove(raw_doc)  # Removes previous accessed raw document from tree to save memory in cache



        self.xml_parser_cache.update({self.doc_path: tree})

        return raw_doc



