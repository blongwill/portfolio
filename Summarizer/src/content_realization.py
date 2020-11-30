#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content realization for multi-document text summarization 
that generates cleaned and compressed versions of sentences."""

__author__ = "Amina Venton, Shannon Ladymon"
__email__ = "aventon@uw.edu, sladymon@uw.edu"

import spacy
import re


def remove_subtree(doc, clean_sent, indices_to_remove_subtree):
    """
    This function removes a subtree given a spaCy Doc object and
    the indices of each token to remove. The indices are used in a
    spaCy Span object to get the tokens of the subtree.

    Note:
        Span object -> slice doc[start : end]
            start: The index of the first token of the span
            end: The index of the first token after the span.

    Args:
        doc: spaCy Doc of the clean sentence
        clean_sent:str the string of the clean sentence Doc object
        indices_to_remove_subtree: list of indices of the subtree to be removed

    Return:
        new_sent:str the newly trimmed sentence

    """

    # Create span of the subtree to be removed
    span_of_subtree_start = indices_to_remove_subtree[0]
    span_of_subtree_end = indices_to_remove_subtree[-1] + 1
    span_to_be_removed = doc[span_of_subtree_start:span_of_subtree_end].text

    # Remove span from the clean sentence
    new_sent = clean_sent.replace(span_to_be_removed, "")

    # Return the new sentence
    return new_sent


def find_subtree_indices(doc, dependency_type):
    """
    This function finds and returns the indices of the entire clause
    (each token) in the subtree to be removed.

    Args:
        doc: spaCy Doc of the clean sentence
        dependency_type:str Options are "appos", "acl", "relcl", "advcl"

    Return:
        indices_to_remove_subtree: list of indices of the subtree

    """

    # List of indices of clause tokens to be removed in the sentence
    indices_to_remove_subtree = []

    # List of unique spaCy hashes for string tokens in the doc
    # Position remains the same from original doc
    hash_ids_of_tokens = [token.orth for token in doc]

    # Iterate through the doc to get the dep clause subtree
    for index, token in enumerate(doc):

        # Check for dependency label
        if token.dep_ == dependency_type:

            # Get the indices of subtree- all tokens of the clause
            for subtree_token in token.subtree:
                # Get the unique hash id for the subtree token
                subtree_token_id = subtree_token.orth

                # Look up the token's index in the doc
                subtree_token_index_in_doc = hash_ids_of_tokens.index(subtree_token_id)

                # Add to list of indices to be removed
                indices_to_remove_subtree.append(subtree_token_index_in_doc)

    # Return list of indices
    return indices_to_remove_subtree


def trim_sentence(doc, dependency_type):

    """
    This function trims a sentence given a spaCy Doc and
    the dependency type corresponding to the clause to be removed.
    The Doc is converted to a string using a spaCy Span object.

    The new sentence with clause removed is returned.
    If no dependency found original clean sentence is returned.

    Note:
        Span object -> slice doc[start : end]
            start: The index of the first token of the span
            end: The index of the first token after the span.

    Args:
        doc: spaCy Doc of the clean sentence
        dependency_type:str Options are "appos", "acl", "relcl", "advcl"

    Return:
        new_sent:str newly trimmed sentence or
        clean_sent:str original clean sentence

    """

    # Convert Doc to string using a Doc Span, whole Span-> no start and end
    clean_sent = doc[:].text

    # Flag that indicates if dependency found, trim the sentence
    dependency_found = any(True for token in doc if token.dep_ == dependency_type)

    if dependency_found:

        # Get list of indices of clause tokens to be removed in the sentence
        indices_to_remove_subtree = find_subtree_indices(doc, dependency_type)

        # Remove the subtree and create new sentence
        new_sent = remove_subtree(doc, clean_sent, indices_to_remove_subtree)

        # Remove any left over punctuation from the subtree removal
        new_sent = clean_punctuation(new_sent)

        # Return the new trimmed sentence
        return new_sent

    # Return original sentence if no dependency found
    return clean_sent


def clean_sentence(original_sent, remove_header, remove_parens, remove_quotes):
    """
    Cleans a sentence by fixing newlines and spaces, and optionally by removing
    the news header, parenthetical information, and unpaired quotes.
   
    Args:
        original_sent: original sentence string
        remove_header: True if the header should be removed from the sentence
        remove_parens: True if parenthetical information should be removed from the sentence
        remove_quotes: True if unpaired quotes should be removed from the sentence

    Returns:
        The cleaned sentence.

    """

    # Start with the clean_sent set to the original as a default
    clean_sent = original_sent



    # Clean sentence by removing the header, if specified
    if remove_header:

        # Matches headers of form "NEW YORK, July 1 (AP) --" and "NEW YORK _ " and "NEW YORK (AP) _"
        header_1_re = re.compile(r"^\s+([A-Z]+\s*)+,?(\s*([a-zA-Z]+\.?\s)*[0-9]*\s*)?(\([a-zA-Z]+\))?\s*((--)|(_))\s*")

        # Matches headers of form "BC-FLA (Tampa) --" and "COLO-SCHOOL-SHOOTING (Littleton, Colo.) _"
        header_2_re = re.compile(r"^\s*([A-Z]+-?)+\s\([a-zA-Z,\s\.]+\)\s((--)|(_))")

        # Matches headers of form "(AP) --" and "_"
        header_3_re = re.compile(r"^(\([a-zA-Z]+\))?\s*((--)|(_))\s*")

        clean_sent = header_1_re.sub(r"", clean_sent)
        clean_sent = header_2_re.sub(r"", clean_sent)
        clean_sent = header_3_re.sub(r"", clean_sent)

    # Clean sentence by removing any parenthetical information, if specified
    if remove_parens:
        # Matches any parenthetical information
        parens_re = re.compile(r"\([^\)]+\)")
        clean_sent = parens_re.sub(r"", clean_sent)

    # Clean sentence by removing any unpaired quotes, if specified
    if remove_quotes:
        # Matches for different types of quotes (", ``, '') to find unpaired quotes
        quote_re = re.compile(r"\"")
        backtick_re = re.compile(r"``")
        apostrophe_re = re.compile(r"''")

        # Get the number of quotes to see if unmatched
        num_quotes = len(quote_re.findall(clean_sent))
        num_backticks = len(backtick_re.findall(clean_sent))
        num_apostrophes = len(apostrophe_re.findall(clean_sent))

        if num_quotes == 1:
            # Remove unpaired " mark
            clean_sent = quote_re.sub(r"", clean_sent)
        if num_backticks == 1 and num_apostrophes == 0:
            # Remove `` (open quote) unpaired with '' (closing quote)
            clean_sent = backtick_re.sub(r"", clean_sent)
        if num_apostrophes == 1 and num_backticks == 0:
            # Remove '' (closing quote) unpaired with `` (open quote)
            clean_sent = apostrophe_re.sub(r"", clean_sent)

    # Remove any extra newlines
    clean_sent = clean_sent.replace("\n", " ").strip()

    # Remove any extra spaces
    spaces_re = re.compile(r" {2,}")
    clean_sent = spaces_re.sub(r" ", clean_sent)

    return clean_sent


def clean_punctuation(sent):
    "Cleans leftover punctuation from sentences after they have been compressed"

    if len(sent) == 0:
        return sent

    # Remove comma pairs with nothing in between
    sent = sent.replace(", , ", " ")

    # Remove extra commas before end of sentence
    sent = sent.replace(", .", ".")

    # Remove extra spaces
    sent = sent.replace("  ", " ")
    sent = sent.replace(" ,", ",")
    sent = sent.replace(" .", ".")

    # Remove commas at start of sentence
    start_comma_re = re.compile(r"^\s*, ")
    sent = start_comma_re.sub(r"", sent)

    # Remove quotations with nothing but a comma inside
    quotes_with_comma_re = re.compile(r"(\"|(``)),(\"|(''))")
    sent = quotes_with_comma_re.sub(r"", sent)

    return sent


def get_compressed_sentences(original_sent, spacy_parser, remove_header, remove_parens, remove_quotes, remove_appos, remove_advcl, remove_relcl, remove_acl):
    """
    This function performs sentence compression given an sentence string and rule-type.
   
    Args:
        original_sent: original sentence string
        spacy_parser: spaCy parser model
        remove_header: True if the header should be removed from the sentence
        remove_parens: True if parenthetical information should be removed from the sentence
        remove_quotes: True if unpaired quotes should be removed from the sentence
        remove_appos: True if appositional modifier should be removed from the sentence
        remove_advcl: True if adverbial clause modifier should be removed from the sentence
        remove_relcl: True if relative clause modifier should be removed from the sentence
        remove_acl: True if a finite or non-finite clausal modifier shoule be removed from the sentence

    Returns:
        sentences_list: list of compressed versions of the original sentence
    """
    # List of sentence strings to return
    sentences_list = []  

    # Get clean version of the sentence
    clean_sent = clean_sentence(original_sent, remove_header, remove_parens, remove_quotes)

    # Add this cleaned version of the sentence to the list
    sentences_list.append(clean_sent)

    # Remove branches of syntax tree from spaCy Doc
    # Sentences will only be trimmed if sent len > 0 and at least one rule condition is met
    if clean_sent and (remove_appos or remove_advcl or remove_relcl or remove_acl):
        doc = spacy_parser(clean_sent)
    
        # Remove appositional modifiers (appos)
        if remove_appos:
            new_sent = trim_sentence(doc, "appos")
            sentences_list.append(new_sent)

        # Remove adverbial clausal modifiers (advcl)
        if remove_advcl:
            new_sent = trim_sentence(doc, "advcl")
            sentences_list.append(new_sent)

        # Remove relative clauses (relcl)
        if remove_relcl:
            new_sent = trim_sentence(doc, "relcl")
            sentences_list.append(new_sent)

        # Remove clausal modifiers (acl)
        if remove_acl:
            new_sent = trim_sentence(doc, "acl")
            sentences_list.append(new_sent)  

    return sentences_list
