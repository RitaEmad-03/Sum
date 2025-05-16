import pandas as pd
import re
import config
from src.functions import extract_ticket_data, _process_dates, extract_closure_info, _get_subsection_pattern, _extract_subsection

from tqdm import tqdm
import os 
import chardet
from functools import partial   # to allow passing an argument to a higher order function
from cleantext import clean
from collections import defaultdict
import spacy
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True)
# Suppress DEBUG and INFO for noisy libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Load the German spaCy model
nlp = spacy.load("de_core_news_md") # de_core_news_sm --  the medium has a better performance
# nlp.add_pipe('sentencizer', config={"punct_chars": [".\n", "!", "?", "-\n"]})


import subprocess
# To be able to run the script locally without modifications
# we need to check if the CUDA GPU is available

def is_cuda_available():
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


if is_cuda_available():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts.prompt import PromptTemplate
    from langchain.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain import hub
    from langchain_community.chat_models import ChatOllama
    from langchain_ollama.llms import OllamaLLM
    
else:
    print("CUDA GPU not detected. Skipping imports.")




def clean_text(text):
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = text.replace("Kurzbeschr (intern):", "").strip()
    # Remove consecutive special characters (excluding ':')
    text = re.sub(r"([^\w\s:().-])\1+", " ", text)
    text = re.sub(r"-{2,100}", " ", text)

    # Remove extra spaces and lines
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

# Working CODE FROM MAY
# def extract_fields_with_keys(patterns, text):
#     extracted_fields = []
#     for key, pattern in patterns:
#         if isinstance(pattern, list):
#             # Handle multiple patterns for a single key
#             matches = []
#             for p in pattern:
#                 match = re.findall(p, text)
#                 if match:
#                     matches.extend(match)
#             if matches:
#                 extracted_fields.append(f"{key}: {' '.join(matches)}")
#         else:
#             match = re.findall(pattern, text)
#             if match:
#                 if key in ["Eskalationskontakt", "Kundenkontakt"]:
#                     extracted_fields.append(
#                         f"{match[0].replace('angelegt:','')} initiiert durch"
#                     )

#                 if key == "newteam":
#                     extracted_fields.append(
#                         f"Das Ticket wird einem neuen Team zugewiesen ({match[0][15:31]}) "
#                     )

#                 elif key == "durch":
#                     extracted_fields.append(f"{match[0].replace(':','')}  mit")

#                 elif key == "Kontaktname":
#                     extracted_fields.append(match[0].replace(":", ""))

#                 elif key == "Grund":
#                     extracted_fields.append(
#                         f"aufgrund einer {match[0].replace(':','')}"
#                     )

#                 elif key in ["Kurzbeschr (intern)", "Kundeninfo"]:
#                     extracted_fields.append(match[0])

#     cleaned_text = (
#         clean_text(" ".join(extracted_fields)) if extracted_fields else clean_text(text)
#     )
#     return cleaned_text


# def extract_fields_with_keys(patterns, text):
#     extracted_fields = []
#     for key, pattern in patterns:
#         if isinstance(pattern, list):
#             # Handle multiple patterns for a single key
#             matches = []
#             for p in pattern:
#                 match = re.findall(p, text)
#                 if match:
#                     matches.extend(match)
#             if matches:
#                 extracted_fields.append(f"{key}: {' '.join(matches)}")
#         else:
#             match = re.findall(pattern, text)
#             if match:
#                 if key in ["Eskalationskontakt"]:#,  "Technikerkontakt"]: , "Kundenkontakt",
#                     extracted_fields.append(
#                         f"{match[0].replace('angelegt:','')} initiiert durch"
#                     )

#                 if key == "newteam":
#                     extracted_fields.append(
#                         f"Das Ticket wird einem neuen Team zugewiesen ({match[0][15:31]}) "
#                     )
#                 # first checks if Kundeninfo is present, if not, check for kurzbeschr (intern)
#                 elif key == "Kundeninfo":
#                     extracted_fields.append(match[0])

#                 elif key == "Kurzbeschr (intern)":
#                     extracted_fields.append(match[0])

#     cleaned_text = (
#         clean_text(" ".join(extracted_fields)) if extracted_fields else clean_text(text)
#     )
#     return cleaned_text


def filter_logs(log):
    """
    The function is designed to filter out logs that contain specific patterns and do not contain "AT000".

    If the initial condition is met, the function processes only the first 4 lines of the log, ignores anything after this.
    
    If the initial condition is not met, the original log is returned unchanged.

    For the first 4 lines of the log, it rechecks after joining the lines. If logs detected, the lines are removed. 

    If no patterns match the first 4 lines, it returns those lines as they are.

    """

    # Check if any of the patterns match the log and "AT000" is not in the log
    if any(pattern.search(log) for pattern in config.log_compiled_patterns) and (
        "AT000" not in log
    ):
        # Split the log into lines and take the first 4 lines

        lines = log.split("\n")[:4]

        # Check if any of the patterns match the first 4 lines when joined together
        if any(
            pattern.search(" ".join(lines)) for pattern in config.log_compiled_patterns
        ):
            # Return the first 4 lines with any matching patterns removed
            return "\n".join(
                line
                for line in lines
                if not any(
                    pattern.search(line) for pattern in config.log_compiled_patterns
                )
            )
        # Return the first 4 lines if no patterns match
        return "\n".join(lines)
    # Return the original log if the initial condition is not met
    return log


def clean_text2(text):
    """
    I have changed the cleaning in this project. I am not sure if I still need to filter out these phrases. However, I am keeping this function for now.
    """

    phrases_to_remove = [
        "Neue AN-Info demarkiert. AN-Info zur Kenntnis genommen.",
        "Neue AN-Info demarkiert AN-Info zur Kenntnis genommen ",
        "AWL: aus",
        "AWL = aus",
        "AN-Info zur Kenntnis genommen",
        "AKI = aus",
        "AKI = ein",
        "wurde AN-Info zur Kenntnis genommen.",
        "wurde AN-Info zur Kenntnis genommen",
        "Neue Technikerinfo demarkiert. Technikerinfo zur Kenntnis genommen.",
        "Neue Technikerinfo demarkiert Technikerinfo zur Kenntnis genommen",
        "Neue Workorder demarkiert Workorder zur Kenntnis genommen",
        "Neue Workorder demarkiert Workorder zur Kenntnis genommen Neue AN-Info demarkiert AN-Info zur Kenntnis genommen",
        "Neue AN-Info: markiert",
        "Neue Workorder: markiert",
        "Neue AN-Info demarkiert AN-Info zur Kenntnis genommen",
        "Neue AN-Info: markiert",
        "Neue AN-Info",
        "Neue AN-Info demarkiert AN-Info zur Kenntnis genommen",
        "Guten Tag,",
        "Neue Workorder:",
        "Neue Workorder de Workorder zur Kenntnis genommen",
        "#IC_OK#",
        "Neue Technikerinfo",
        "Neue Workorder de Workorder zur Kenntnis genommen",
        "   Neue Workorder demarkiert.\n   Workorder zur Kenntnis genommen.\n   \n   ",
        "¿",
        "   Neue Workorder: markiert.\n   \n   ",
        "Neue Workorder demarkiert.",
        "markiert",
        "Workorder zur Kenntnis genommen",
    ]

    patterns = [
        #  r'(?:neuer (?:Technikerkontakt|Kundenkontakt) initiiert durch .*? mit .*? aufgrund einer \w+)',
        # r'(?:neuer (?:Technikerkontakt|Kundenkontakt) initiiert durch .*? mit aufgrund einer \w+)',
        "Neue Workorder de Workorder zur Kenntnis genommen",
        r"(?:Neue Technikerinfo: markiert\.|Neue Technikerinfo demarkiert\. Technikerinfo zur Kenntnis genommen\.|Neue Technikerinfo demarkiert\.|Neue Kundeninfo demarkiert\. Kundeninfo zur Kenntnis genommen\. Neue Technikerinfo demarkiert\. Technikerinfo zur Kenntnis genommen\.)",
        # r'^Lösungskoordinator:.*', 'Neue Workorder demarkiert.',
        r"Neue Kundeninfo demarkiert. Kundeninfo zur Kenntnis genommen.",
        "Neue AN-Info:",
        #   r'^Lösungskoordinator.*',
        "Neue AN-Info demarkiert (angezeigt)",
        "Neue AN-Info demarkiert.",
        "Neue Workorder:",
        "Neue Workorder de Workorder zur Kenntnis genommen",
        "   Neue Workorder demarkiert.\n   Workorder zur Kenntnis genommen.\n   \n   ",
        "   Neue Workorder: markiert.\n   \n   ",
        "Workorder zur Kenntnis genommen",
        "¿",
    ]

    # Remove phrases
    pattern = r"\s*Neue Workorder.*?\.?\n\s*"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text, flags=re.DOTALL)
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")

    # Remove patterns
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL) # re.MULTILINE: Allows ^ and $ to match the start and end of each line. re.DOTALL: Allows . to match any character, including newlines.
#  # Clean up the text
    text = text.rstrip("\n")  # Remove trailing newline characters
    text = text.replace(".", " ")  # Replace dots with spaces
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with one and strip whitespace
    text = re.sub(r":", " ", text).strip()  # Replace colons with spaces and strip whitespace
    text = re.sub(r"_{2,}", " ", text).strip()  # Replace multiple underscores with one space and strip whitespace

    return text


def split_into_sentences(paragraph):
    """
    Split a paragraph into sentences using spaCy.
    
    Parameters:
        paragraph (str): The paragraph to split.
        
    Returns:
        list: List of sentences.
    """
    doc = nlp(paragraph) if paragraph else nlp("")
    return [sent.text.strip() for sent in doc.sents]

# def is_human_readable(text):
#     """
#     Check if text contains human-readable content based on NLP rules.
#     Now handles both single sentences and paragraphs.
    
#     Parameters:
#         text (str): The text to analyze.
        
#     Returns:
#         bool: True if the text is human-readable, False otherwise.
#     """
#     # First check if this is an Auftrag entry
#     if "AT000" in text:
#         return True
        
#     # Split into sentences if it's a paragraph
#     sentences = split_into_sentences(text)
    
#     # If no valid sentences found, return False
#     if not sentences:
#         return False
        
#     # Check if at least one sentence meets our criteria
#     for sentence in sentences:
#         doc = nlp(sentence)
#         print(f"{doc = }")
#         # Rule 1: Sentence must have at least one verb
#         has_verb = any(token.pos_ == "VERB" for token in doc)
        
#         # Rule 2: Sentence must have a minimum length (e.g., at least 3 words)
#         min_length = len([token for token in doc if not token.is_punct]) >= 2
        
#         # Rule 3: Sentence should not contain excessive special characters or codes
#         special_char_count = sum(1 for token in doc if not token.is_alpha and not token.is_punct)
#         total_tokens = len([token for token in doc if not token.is_punct])
#         if total_tokens == 0:  # Prevent division by zero
#             excessive_special_chars = True
#         else:
#             excessive_special_chars = (special_char_count / total_tokens) > 0.3

#         # Combine rules
#         if has_verb and min_length and not excessive_special_chars:
#             return True

#     return False


def is_human_readable(sentence):
    """
    Check if a sentence is human-readable based on NLP rules.
    """

    doc = nlp(sentence)
    
    # Rule 1: Sentence must have at least one verb or adjective AND one noun or proper noun
    has_verb_or_adj = any(token.pos_ in {"VERB", "ADJ", "AUX", "ADV"} for token in doc) # AUX is auxiliary verb
    has_noun_or_propn = any(token.pos_ in {"NOUN", "PROPN"} for token in doc)

    # Rule 2: Sentence must have a minimum length (e.g., at least 2 words)
    min_length = len([token for token in doc if not token.is_punct]) >= 2
    
    # Rule 3: Sentence should not contain excessive special characters or codes
    special_char_count = sum(1 for token in doc if not token.is_alpha and not token.is_punct)
    total_tokens = len([token for token in doc if not token.is_punct])
    if total_tokens == 0:  # Prevent division by zero
        excessive_special_chars = True
    else:
        excessive_special_chars = (special_char_count / total_tokens) > 0.6  # More than 60% special chars
    logging.debug(f"{doc = }")
    logging.debug(f"{excessive_special_chars = }")
    logging.debug(f"{has_verb_or_adj = }")
    logging.debug(f"{has_noun_or_propn = }")
    logging.debug(f"{min_length = }")

    result = has_verb_or_adj and has_noun_or_propn and min_length and not excessive_special_chars

    # if result:
    #     print(f"{sentence = }")
    #     print(f"{has_verb_or_adj = }")
    #     print(f"{min_length = }")
    #     print(f"{excessive_special_chars = }")
        
        
    # Combine rules
    return result

# def split_text_punctuation(text):
#     """Splits text based on the provided punctuation pattern."""
#     punct_chars = [r"\.\n", r"!", r"\?", r"-\n"]
#     pattern = "(" + "|".join(punct_chars) + ")"
#     return [item.strip() for item in re.split(pattern, text) if item.strip()]

def filter_terminal_commands_output(text):
    """
    filter out terminal commands and outputs from the text.
    """
    lines = text.splitlines()
    filtered_lines = []

    for line in lines:
        if not is_terminal_command(line):
            filtered_lines.append(line)
        else:
            logging.info(f"Terminal command: {line}")
            logging.info(f"filtered_lines: {filtered_lines}")
            break
    
    return "\n".join(filtered_lines)


# def remove_all_line_special_chars(text):
#     """
#     If a line has no text in it, it shall be removed 
#     """
#     lines = text.splitlines()
#     filtered_lines = []

#     for line in lines:
#         doc = nlp(line)

#         alpha_char_count = sum(1 for token in doc if token.is_alpha)

#         if alpha_char_count: 
#             filtered_lines.append(line)
#         else: 
#             logging.debug(f"Omitted line special char - {line}")
    
#     return "\n".join(filtered_lines)

def remove_all_line_special_chars(text):
    """
    Removes lines with no alphabetical characters.
    """
    lines = text.splitlines()
    filtered_lines = [line for line in lines if re.search(r'[a-zA-Z]', line)]
    
    for line in (set(lines) - set(filtered_lines)):
        logging.debug(f"Omitted line special char - {line}")
    
    return "\n".join(filtered_lines)

def filter_text_DOCSIS(text):
    """
    Filters text, removing lines matching patterns and subsequent lines until a single newline.

    Args:
        input_text: The input text as a string.

    Returns:
        The filtered text as a string.
    """
    lines = text.splitlines()
    filtered_lines = []
    skip_mode = False

    for line in lines:
        if skip_mode:
            if not line.strip():
                skip_mode = False
            else:
                continue
        if "DOCSIS-Status" in line or re.match(r"\s{5,}-", line):
            skip_mode = True
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def filter_human_readable(paragraphs):
    """
    Filter paragraphs to keep only human-readable ones.
    Handles both single strings and lists of strings.
    
    Parameters:
        paragraphs (str or list): Text to filter, either a single string or list of strings
        
    Returns:
        list: List of human-readable paragraphs
    """
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    
    readable_paragraphs = []

    for paragraph in paragraphs:

        # First check if this is an Auftrag entry
        if "AT000" in paragraph:
            readable_paragraphs.append(clean(paragraph, lower=False))
            continue
        
        # filtered_lines = [line for line in paragraph.splitlines() if is_human_readable(line)]

        # Split the paragraph into lines based on a period followed by a newline
        filtered_lines = [line for line in re.split(r"\. ?\n", paragraph) if is_human_readable(line)]
        logging.debug(f"{filtered_lines = }")
        # return [sentence.strip() for sentence in re.split(r"\. ?\n", paragraph) if sentence.strip()]
        

        # Join the filtered lines back into a paragraph
        filtered_paragraph = "\n".join(filtered_lines)

        cleaned_paragraph = clean_whitespace(filtered_paragraph)

        # Filter out non-human-readable lines
        # filtered_lines = [line for line in split_text_punctuation(paragraph) if is_human_readable(line)]
        # for line in filtered_lines:
        #     logging.info(f"{line = }")

        # # Join the filtered lines back into a paragraph

        # filtered_paragraph = "\n".join(filtered_lines)

        logging.info(f"{cleaned_paragraph = }")
        sentences = split_into_sentences(cleaned_paragraph)
        for sent in sentences:
            logging.debug(f"{sent = }")
            if is_human_readable(sent):
                logging.debug(f"{sent = }")
                readable_paragraphs.append(sent)
            
    return readable_paragraphs

# def filter_human_readable(text):
#     """
#     Filter out sentences that don't meet the human-readable criteria.
#     """
#     readable_paragraphs = []
#     # Split into sentences
#     sentences = split_into_sentences(text)
    
#     # Filter sentences
#     filtered_sentences = [sentence for sentence in sentences if is_human_readable(sentence)]
    
#     # Reconstruct the text (optional)
#     joined_sent =  " ".join(filtered_sentences)
#     readable_paragraphs.append(joined_sent)
#     return readable_paragraphs



def _remove_useless_sections(df, keep_technikerkontakt=False):
    """
    Filters out rows from the DataFrame based on specific conditions related to the 'benutzer' and 'worklog' columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'benutzer' and 'worklog' columns.
    remove_technikerkontakt (bool): If True, removes rows containing "neuer Technikerkontakt". Default is True.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Substrings to filter out for 'AR_SERVER' logic
    # ar_server_substrings = ["Auftrag", "Von TTWOS beantwortet"]  # I am keeping the upper case

    # Substrings to filter out regardless of 'benutzer'
    general_substrings = ["AN-Info", "Neue Kundeninfo", "Neue Technikerinfo", "neuer Technikerkontakt"]

    # Add "neuer Technikerkontakt" to the general substrings if the argument is True
    if keep_technikerkontakt:
        general_substrings.remove("neuer Technikerkontakt")

    # Filter out rows where 'benutzer' is "Distributed Server" or "AR_SERVE" (regardless of 'worklog')
    df_filtered = df[
        ~df["Benutzer"].isin(["Distributed Server", "AR_SERVE", "hrobot2"])
    ]

    # Filter out rows where 'benutzer' is 'AR_SERVER' and 'worklog' doesn't contain any of the AR_SERVER substrings
    df_filtered = df_filtered[
        ~(
            (df_filtered["Benutzer"] == "AR_SERVER") &
            ~(df_filtered["BearbLog"].str.contains("Auftrag AT00")) #Auftrag AT
        )
    ]

    # Filter out rows where 'worklog' contains any of the general substrings
    df_filtered = df_filtered[
        ~df_filtered["BearbLog"].str.contains("|".join(general_substrings))
    ]

    return df_filtered


# Working CODE FROM MAY
# def final_processing(df_explode):
#     df_explode["Datum"] = pd.to_datetime(df_explode["Datum"], format="%d.%m.%Y%H:%M:%S")
#     df_explode["day"] = df_explode["Datum"].dt.strftime("%d-%m-%Y")
#     def_users = ["AR_SERVE", "AR_SERVER", "Distributed Server"]
#     df_explode["processed_bearblog"] = df_explode.processed_bearblog.apply(
#         lambda x: clean_text2(x)
#     )    
# #         # Combine the "AT0000", "Auftrag", "Von TTWOS beantwortet" conditions into a single regex pattern
# #     regex_pattern = "|".join(["AT0000", "Auftrag", "Von TTWOS beantwortet"])

# #     # Apply the filtering
# #     df_explode = df_explode[
# #         # Condition 1: `processed_bearblog` contains any of "AT0000", "Auftrag", or "Von TTWOS beantwortet"
# #         (df_explode["processed_bearblog"].str.contains(regex_pattern))]

#     df_explode = df_explode[
#         # Condition 2: Exclude rows where `Benutzer` is in `def_users` OR `processed_bearblog` contains "Neue Kundeninfo"
#         (
#             (~df_explode.Benutzer.isin(def_users)) & 
#             (~df_explode["processed_bearblog"].str.contains("Neue Kundeninfo")) &
#             (~df_explode["processed_bearblog"].str.contains("neuer Technikerkontakt"))
#         )
#     ]
    
#     combined_bearblogs = []

#     for ticketnr, group_df in df_explode.groupby("Ticketnr"):
#         i = 0
#         combined_bearblog = ""
#         for row in group_df.itertuples():
#             if row.processed_bearblog in [""]:
#                 continue
#             else:
#                 i += 1
#                 combined_bearblog += f"{row.day}: {row.processed_bearblog}\n"

#         combined_bearblogs.append(combined_bearblog.strip())
#     new_df = pd.DataFrame()
#     new_df["Ticketnr"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first").Ticketnr.values
#     )
#     new_df["bearblog"] = combined_bearblogs
#     new_df["exakte_Problem"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first").exakte_Problem.values
#     )
#     new_df["Produkt"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["dienst_produkt"].values
#     )
#     new_df["Technik"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["Technik"].values
#     )
#     new_df["Problembeschr"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["Problembeschr"].values
#     )

#     return new_df, df_explode

# May working code
# def final_processing(df_explode):
    
#     combined_bearblogs = []

#     for ticketnr, group_df in df_explode.groupby("Ticketnr"):
#         i = 0
#         combined_bearblog = ""
#         for row in group_df.itertuples():
#             if row.processed_bearblog in [""]:
#                 continue
#             else:
#                 i += 1
#                 combined_bearblog += f"{row.day}: {row.processed_bearblog}\n"

#         combined_bearblogs.append(combined_bearblog.strip())
#     new_df = pd.DataFrame()
#     new_df["Ticketnr"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first").Ticketnr.values
#     )
#     new_df["bearblog"] = combined_bearblogs
#     new_df["exakte_Problem"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first").exakte_Problem.values
#     )
#     new_df["Produkt"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["dienst_produkt"].values
#     )
#     new_df["Technik"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["Technik"].values
#     )
#     new_df["Problembeschr"] = list(
#         df_explode.drop_duplicates("Ticketnr", keep="first")["Problembeschr"].values
#     )

#     return new_df, df_explode

def process_bearblog(cleaned_bearblog):
    combined_bearblog = ""
    # Skip if cleaned_bearblog is an empty list
    if not cleaned_bearblog:
        return ""

    # Handle lists or lists of lists
    if isinstance(cleaned_bearblog, list):
        # Flatten the list if it's a list of lists
        flattened_list = []
        for item in cleaned_bearblog:
            if isinstance(item, list):
                flattened_list.extend(item)
            else:
                flattened_list.append(item)

        # Join the flattened list with a single dot separator

        if flattened_list and (flattened_list[-1].endswith('.')) :
            bearblog_text = ' '.join(flattened_list)
        else:
            bearblog_text = '. '.join(flattened_list)

    # Strip trailing whitespace
    combined_bearblog = bearblog_text.strip()

    return combined_bearblog




# def process_bearblog(df):
#     combined_bearblog = ""

#     for row in df.itertuples():
#         # Skip if cleaned_bearblog is an empty list
#         if not row.cleaned_bearblog:
#             continue

#         # Handle lists or lists of lists
#         if isinstance(row.cleaned_bearblog, list):
#             # Flatten the list if it's a list of lists
#             flattened_list = []
#             for item in row.cleaned_bearblog:
#                 if isinstance(item, list):
#                     flattened_list.extend(item)
#                 else:
#                     flattened_list.append(item)

#             # Join the flattened list with a single dot separator
#             bearblog_text = ". ".join(flattened_list)

#             # Add the date prefix
#             combined_bearblog += f"{row.day}: {bearblog_text}\n"

#     # Strip trailing whitespace
#     combined_bearblog = combined_bearblog.strip()

#     # Add the combined bearblog as a new column to the existing DataFrame
#     df["processed_bearblog"] = combined_bearblog

#     return df


def _find_matching_pattern(text, section_patterns):
    """
    Finds a match in the text using any of the patterns in the list.
    Also checks for Eskalationskontakt pattern.
    
    Parameters:
        text (str): The text to search within.
        pattern_list (list of re.Pattern): The compiled regex patterns to search for.
        
    Returns:
        tuple: (bool, bool) - (pattern_match, has_escalation)
    """
    has_pattern = False
    has_escalation = False
    
    for pattern, regex in section_patterns:
        if isinstance(regex, list):
            for r in regex:
                print(f"Multiple matches: {r}")
                if r.search(text):
                    has_pattern = True
                    if pattern == "Eskalationskontakt":
                        has_escalation = True
        else:                
            if regex.search(text):
                has_pattern = True
                if pattern == "Eskalationskontakt":
                    has_escalation = True
                    
    return has_pattern, has_escalation


def extract_subsection_info(benutzer, bearblog):
    """
    Processes the 'BearbLog' text based on the 'Benutzer' value and extracts relevant information.
    
    Parameters:
        benutzer (str): The value from the 'Benutzer' column.
        bearblog (str): The text from the 'BearbLog' column.
        
    Returns:
        list or str: The extracted information based on the conditions specified.
    """
    extracted_info = []

    # if "Neuer Auftrag" in bearblog:
    #     auftrag_num = extract_auftrag_number(bearblog)
    #     if auftrag_num:
    #         extracted_info.append(
    #             f"Ein neuer auftrag wurde angelegt zugewiesen mit {auftrag_num}"
    #         )

    if (benutzer == "AR_SERVER") or ("Auftrag AT00" in bearblog): # benutzer == "AR_SERVER" this assumes that AR_SERVER has the pattern "Auftrag AT00" due to previous filtering
        extract_auftrag_reply_details(bearblog, extracted_info)
    
    elif _find_matching_pattern(bearblog, config.section_patterns)[0]: # extracting the first output of the tuple
        has_pattern, has_escalation = _find_matching_pattern(bearblog, config.section_patterns)
        
         # Rest of the existing pattern matching code
        kundeninfo_pattern = _get_subsection_pattern("Kundeninfo", config.subsection_patterns)
        kurzbeschr_pattern = _get_subsection_pattern("Kurzbeschr (intern)", config.subsection_patterns)

        kundeninfo = _extract_subsection(bearblog, kundeninfo_pattern)
        kurzbeschr = _extract_subsection(bearblog, kurzbeschr_pattern)

        if has_escalation:
            extracted_info.append("Eine neue eskalation wurde initiert, da ")
            logging.info("Eine neue eskalation wurde initiert, da ")
            logging.info(f"{kurzbeschr = }")
        if kundeninfo:
            logging.debug(f"Kundeninfo: {kundeninfo}")
            kundeninfo = kundeninfo.replace("Kundeninfo:", "Kunde kontaktiert, ")
            logging.debug(f"Kundeninfo: {kundeninfo}")
            cleaned_kundeninfo = clean_whitespace(kundeninfo)
            paragraphs_human_readable = remove_logs_terminal_tbls_from_paragraphs(cleaned_kundeninfo)
            # print(f"{paragraphs_human_readable}")
            extracted_info.append(paragraphs_human_readable)
            # print(f"{extracted_info = }")
        elif kurzbeschr:
            logging.info(f"Kurzbeschr: {kurzbeschr}")
            kurzbeschr = kurzbeschr[21:] # remove the "Kurzbeschr (intern):" part
            logging.info(f"{kurzbeschr = }")
            paragraphs_human_readable = remove_logs_terminal_tbls_from_paragraphs(kurzbeschr)
            extracted_info.append(paragraphs_human_readable)
        else: 
            logging.info("Within a main section, No kundeninfo or kurzbeschr found")
    
    # If no specific patterns matched, split into paragraphs
    if not extracted_info:
        # Split on two or more newlines, even if there is whitespace between them
        paragraphs_human_readable = remove_logs_terminal_tbls_from_paragraphs(bearblog)
        return paragraphs_human_readable
    else:
        return extracted_info 

def remove_logs_terminal_tbls_from_paragraphs(text):

    text_no_terminal = filter_terminal_commands_output(text) 

    text_no_terminal_DOCSIS = filter_text_DOCSIS(text_no_terminal)

    text_no_terminal_DOCSIS_special_chars = remove_all_line_special_chars(text_no_terminal_DOCSIS)

    paragraphs_no_terminal = re.split(r'\n\s*\n', text_no_terminal_DOCSIS_special_chars)

    logging.debug(f"paragraphs_no_terminal: {paragraphs_no_terminal}")
    # Strip whitespace from each paragraph and filter out empty strings
    paragraphs_no_logs_terminal = [filter_logs(p) for p in paragraphs_no_terminal if p.strip()]
    logging.debug(f"paragraphs_no_logs_terminal: {paragraphs_no_logs_terminal}")
    # paragraphs_no_logs_terminal = [p.strip() for p in paragraphs_no_logs if p.strip() and not is_terminal_command(p)]  
    
    #  if p.strip() and not is_terminal_command(p)  

    logging.debug(f"Paragraphs_no_logs_terminal: {paragraphs_no_logs_terminal}")
    paragraphs_no_logs_terminal_tbls = [remove_table_lines(p) for p in paragraphs_no_logs_terminal if p.strip()]

    logging.debug(f"paragraphs_no_logs_terminal_tbls: {paragraphs_no_logs_terminal_tbls}")
    paragraphs_human_readable = filter_human_readable(paragraphs_no_logs_terminal_tbls)
    logging.debug(f"Paragraphs_human_readable: {paragraphs_human_readable}")
    return paragraphs_human_readable   

def detect_assigned_entity_in_auftrag(text):
    """
    Detects if any of the keys from the auftrag_mapping are present in the given text.

    :param text: The input text to search within.
    :return: the 
    """
    # Find all keys that are present in the text
    matching_keys = [key for key in config.auftrag_mapping.keys() if key in text]
    
    # If exactly one key is found, return its corresponding value
    if len(matching_keys) == 1:
        return config.auftrag_mapping.get(matching_keys[0], None)
    else:
        return None    


def extract_auftrag_reply_details(bearblog, extracted_info):
    """
    Extracts the 'Auftrag', 'Ruckmeldebeschreibung', 'Losungs' and 'Abschlussmeldung' from the 'BearbLog' text.
    """

    # Extract the assigned person
    assigned_entity = detect_assigned_entity_in_auftrag(bearblog)

    # extract the full line for auftrag
    auftrag_line = None
    for line in bearblog.splitlines():
        if "Auftrag AT00" in line:
            auftrag_line = line
            break
        
        # Append the full line to the extracted_info list
    if auftrag_line:
        if assigned_entity:
            # auftrag_line.replace("\.", "")
            auftrag_line = auftrag_line.replace(".", "")  # Remove the full stop
            auftrag_with_assigned = f"{auftrag_line} von {assigned_entity}"
            extracted_info.append(auftrag_with_assigned)
        else:
            extracted_info.append(auftrag_line)
    
    # Extract 'Losungs'
    losungs_pattern = _get_subsection_pattern("Losungs", config.subsection_patterns)
    if losungs_pattern:
        losungs = _extract_subsection(bearblog, losungs_pattern)
            # print(f"{losungs = }")
        if losungs:
            extracted_info.append(clean_whitespace(losungs))
        
        # Extract 'Ruckmeldebeschreibung'
    ruckmelde_pattern = _get_subsection_pattern("Ruckmeldebeschreibung", config.subsection_patterns)
    if ruckmelde_pattern:
        ruckmelde = _extract_subsection(bearblog, ruckmelde_pattern)
        if ruckmelde:
            extracted_info.append(clean_whitespace(ruckmelde))

                # Extract 'Ruckmeldebeschreibung'
    abschluss_pattern = _get_subsection_pattern("Abschlussmeldung", config.subsection_patterns)
    if abschluss_pattern:
        abschlussmeldung = _extract_subsection(bearblog, abschluss_pattern)
        if abschlussmeldung:
            extracted_info.append(clean_whitespace(abschlussmeldung))

    # print(f"{extracted_info = }")

# # working code, doesn't account for sd int des
# def is_terminal_command(text):
#     # Define valid terminal commands
#     valid_commands = {
#         "ls", "cd", "mkdir", "rm", "git", "ping", "curl", "ps", "chmod", 
#         "scp", "ssh", "sudo", "python", "sh"
#     }

#     # Define terminal prompt symbols
#     prompt_symbols = r"[\$#>@]"

#     # Define patterns to detect terminal commands
#     command_patterns = [
#         rf"^\s*{prompt_symbols}.*",  # Terminal prompt at the start
#         rf"\b({'|'.join(valid_commands)})\b.*",  # Valid command anywhere
#         rf"^\s*\[.*\]\s*{prompt_symbols}.*",  # Prompt with username and hostname
#     ]

#     for pattern in command_patterns:
#         if re.search(pattern, text):  # Use re.search to find patterns anywhere in the string
#             return True
#     return False


# def is_terminal_command(text):
#     """
#     Detect terminal command prompts and router logs.
#     """
#     # Detect terminal command prompts and router logs
#     command_patterns = [
#         r"^\s*(\$|#|>|@).*",   # Terminal prompt
#         r"^\s*\[.*\]\s*(\$|#|>|@).*",  # Prompt with username/hostname
#         r"^\s*(ls|cd|mkdir|rm|git|ping|curl|ps|chmod|scp|ssh|sudo|python|sh)\b.*",  # Common commands
#         r"^\s*%.*?-\d+-.*?: .*",  # Router logs like %DIALER-6-BIND
#         r".*?-\d+-.*?: .*",       # Alternative router log format
#         r"^.*?:\s*%\w+(-\d+)?-\w+:.*"  # Another router log variant
#     ]

#     return any(re.search(pattern, text) for pattern in command_patterns)

def is_terminal_command(text):
    """
    Detect terminal command prompts, router logs, and network device CLI commands.
    """
    # Define networking CLI patterns
    networking_commands = {
        "sh", "show", "int", "interface", "des", "description",    # Interface commands    
        "ip", "route", "nat", "access-list",                       # IP commands                       
        "conf", "configure", "terminal",                           # Config commands                           
        "debug", "undebug", "logging",                            # Debug commands                            
        "clear", "copy", "write", "erase",                        # System commands                        
        "ping", "traceroute", "telnet", "ssh"                     # Connectivity commands                     
    }

    # Define terminal command patterns
    command_patterns = [
        # Terminal prompts
        # r"^\s*(\$|#|>|@).*",    # this line causes exclusion of useful information                                  
        r"^\s*\[.*\]\s*(\$|#|>|@).*",                            
        r"^\s*(ls|cd|mkdir|rm|git|ping|curl|ps|chmod|scp|ssh|sudo|python|sh)\b.*",
        
        # Router logs
        r"^\s*%.*?-\d+-.*?: .*",                                  
        r".*?-\d+-.*?: .*",                                       
        r"^.*?:\s*%\w+(-\d+)?-\w+:.*",                           
        
        # Network device prompts 
        r"^(Router|Switch|Firewall|[\w-]+device)[>#].*",          # Only match specific device types
        r".*#\s*sh\s+.*",                                         
        
        # Networking commands - only match if preceded by prompt

        # rf"^[\w-]+[>#]\s*({'|'.join(networking_commands)})\b\s*(.*)?$"     
        rf"^([\w-]+)?[>#]\s*({'|'.join(networking_commands)})\b\s*(.*)?$" 
        # rf"^[\w-]+[>#]\s*({'|'.join(networking_commands)})\b.*"   # Commands must follow a prompt

    ]

    return any(re.search(pattern, text.strip(), re.IGNORECASE) for pattern in command_patterns)

def remove_table_lines(text):
    """
    Removes lines that appear to be part of tables or formatted logs.
    """
    # Define patterns for table-like content
    table_patterns = [
        r'\s{2,}',  # Multiple whitespaces
        r'^\s*\d{2,}\s+\w+\s+\d{2}:\d{2}:\d{2}:.*',  # Timestamped logs
        r'^\s*\d{2}\s+\d{2}:\d{2}:\d{2}\s+.*',  # Another timestamp format
        r'^.*\d{2}:\d{2}:\d{2}.*%.*',  # Router logs with timestamps
    ]
    
    # Split the text into lines and filter
    lines = text.splitlines()
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            filtered_lines.append(line)
            continue
            
        # Skip if line matches any table pattern
        if any(re.search(pattern, line) for pattern in table_patterns):
            continue
            
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


## Working CODE FROM MAY
# def starter(file_content, key, ticket_required_keys):
#     df = extract_ticket_data(file_content, key, ticket_required_keys)

#     df = df.rename(columns={"desc": "exakte_Problem", "produkt": "dienst_produkt"})

#     df_explode = df.explode(["Datum", "Benutzer", "BearbLog"])
#     clean_whitespace(df_explode)
#     df_explode = df_explode.sort_values(by=["Ticketnr", "Datum"])
#     df_explode["processed_bearblog"] = df_explode.BearbLog.apply(
#         lambda x: clean_text(extract_fields_with_keys(config.patterns, x))
#     )
#     df_explode["processed_bearblog"] = df_explode["processed_bearblog"].apply(
#         filter_logs
#     )
#     df_explode["new_processed_bearblog"] = df_explode.processed_bearblog.apply(
#         lambda x: clean_text2(x)
#     )
#     df_explode["dienst_produkt"] = df_explode["dienst_produkt"].apply(
#         lambda x: x.replace(":", "")
#     )
#     df_explode["Benutzer"] = df_explode["Benutzer"].apply(lambda x: x.replace("\n", ""))
#     return df_explode


def clean_text_from_library(text):
    # customized function from clean-text library
    cleaned_text = clean(text, lower=False,  
      no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_punct="",          # instead of removing punctuations you may replace them
    replace_with_url="",
    replace_with_email="",
    replace_with_phone_number="",
    replace_with_number="",
    replace_with_currency_symbol="",
    lang="de"                       # set to 'de' for German special handling
)
    # removing new lines 
    return cleaned_text.replace("\n", "")

def _preprocess_dataframe(df):
    """
    Performs initial preprocessing of the dataframe including renaming columns and exploding data.
    
    Args:
        df (pd.DataFrame): Input dataframe with raw ticket data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.rename(columns={"desc": "exakte_Problem", "produkt": "dienst_produkt"})
    df_explode = df.explode(["Datum", "Benutzer", "BearbLog"])
    return df_explode


# def remove_table_lines(text):
#     """
#     Removes all lines that are part of tables from the given text.
    
#     Parameters:
#         text (str): The text containing potential tables.
        
#     Returns:
#         str: The text with all table lines removed.
#     """
#     # Define a pattern to find lines with multiple whitespace separations
#     table_pattern = re.compile(r'\s{2,}') # two or more whitespaces
    
#     # Split the text into lines and filter out table lines
#     non_table_lines = []
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             non_table_lines.append(line)
#             continue
#         # If the line does not match the table pattern, keep it
#         if len(table_pattern.findall(line)) <= 1:
#             non_table_lines.append(line)
    
#     # Join the non-table lines back into a single string
#     return '\n'.join(non_table_lines)

def _clean_text_columns(df):
    """
    Cleans text in specific columns of the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with cleaned text columns
    """
    df["Benutzer"] = df["Benutzer"].apply(lambda x: clean(x, lower=False))
    df["dienst_produkt"] = df["dienst_produkt"].apply(lambda x: x.replace(":", ""))
    # df["exakte_Problem"] = df["exakte_Problem"].apply(lambda x: clean_text_from_library(x))
    return df

# Global dictionary to store auftrag-entity mappings
auftrag_entity_map = defaultdict(str)

def extract_auftrag_number(text):
    """Extract AT00* number from text."""
    # Handle both quoted and unquoted AT00* numbers
    match = re.search(r"['\"]?(AT00\d+)['\"]?", text)
    return match.group(1) if match else None

def update_auftrag_entity_mapping(bearblog):
    """Update the global auftrag-entity map."""
    # For "Auftrag AT00*" pattern
    if "Auftrag AT00" in bearblog:
        auftrag_num = extract_auftrag_number(bearblog)
        if auftrag_num:
            entity = detect_assigned_entity_in_auftrag(bearblog)
            if entity:
                auftrag_entity_map[auftrag_num] = entity

# Update assignment messages with entity information
def update_assignment_message(text):
    
    if "Neuer Auftrag" in text: 

        
        # Find the starting index of "Problembeschreibung:"
        
        start_index = text.find("Problembeschreibung:")
    
        # If "Problembeschreibung:" is found, extract the text from that point
        auftrag_problem = text[start_index:] if start_index != -1 else None
        
        match = re.search(r"Neuer Auftrag ['\"]?(AT00\d+)['\"]?", text)
        if match:
            auftrag_num = match.group(1)
            # print(f"{auftrag_num = }")
            entity = auftrag_entity_map[auftrag_num] 
            # print(auftrag_entity_map[auftrag_num])
            if auftrag_num and entity:
                return f"Ein neuer auftrag wurde angelegt zugewiesen {entity}. {auftrag_problem}"
        else:
            return auftrag_problem
            
    return text

        # entity = detect_assigned_entity_in_auftrag(bearblog)
        # print(f"{auftrag_num = }")
        # if entity:
        #     auftrag_entity_map[auftrag_num] = entity

    # if "Ein neuer auftrag wurde angelegt zugewiesen" in text:
    #     # Find the associated AT00 number from the original bearblog
    #     auftrag_num = next((num for num in auftrag_entity_map.keys() if num in text), None)
    #     if auftrag_num and auftrag_entity_map[auftrag_num]:
    #         return text + f" an {auftrag_entity_map[auftrag_num]}"
    # return text


def remove_text_within_symbols(text):
  """
  Removes specified symbols from a given text string.
  """
  return re.sub(r"#+.*?#+", "", text) 

def _process_bearblog_entries(df):
    """
    Processes BearbLog entries in the dataframe.
    """
    df["BearbLog"] = df["BearbLog"].apply(replace_abbreviations)

    df["cleaned_bearblog"] = df.apply(
        lambda row: extract_subsection_info(row['Benutzer'], row['BearbLog']),
        axis=1
    )
    
    # Update auftrag-entity mappings
    for _, row in df.iterrows():
        update_auftrag_entity_mapping(row['BearbLog'])
    
    df["processed_bearblog"] = df['cleaned_bearblog'].apply(process_bearblog)
    
    df["processed_bearblog"] = df["processed_bearblog"].apply(update_assignment_message)
    
    df = df.query('processed_bearblog != ""')
    df["processed_bearblog"] = df['day'] + ": " + df['processed_bearblog']
    df["processed_bearblog"] = df['processed_bearblog'].apply(remove_text_within_symbols)

    return df

def clean_process_bearblog(file_content, key, ticket_required_keys):
    """
    Main function to clean and process ticket BearbLog data.
    
    Args:
        file_content (str): Raw file content containing ticket data
        key (str): Key for extracting ticket data
        ticket_required_keys (list): List of required keys for ticket data
        
    Returns:
        tuple: (Processed dataframe, Original number of log entries)
    """
    # Extract initial data
    df = extract_ticket_data(file_content, key, ticket_required_keys)
    
    # Initial preprocessing
    df_explode = _preprocess_dataframe(df)
    len_logs = len(df_explode)
    
    # Process dates and sort
    df_explode = _process_dates(df_explode)
    df_explode = df_explode.sort_values(by=["Ticketnr", "Datum"])
    
    # Clean text columns
    df_explode = _clean_text_columns(df_explode)
    
    # Remove useless sections
    df_explode = _remove_useless_sections(df_explode)
    
    # Process BearbLog entries
    df_explode = _process_bearblog_entries(df_explode)
    
    # print(f"{auftrag_entity_map = }")

    return df_explode, len_logs
    


def replace_abbreviations(text):
    """
    Replaces known abbreviations in the given text with their full forms, ensuring abbreviations are treated as separate words.

    Parameters:
    text (str): The input text containing abbreviations.

    Returns:
    str: The text with abbreviations replaced by their full forms.
    """
    # Dictionary mapping abbreviations to their full forms
    abbreviations = {
        r"\bAsp.\b": "Ansprechpartner",
        r"\bASP\b": "Ansprechpartner",
        r"\bRTR\b": "Router",
        r"\bBC\b": "Basic checks",
        r"\bn.e.\b": "nicht erreichbar",
        r"\bKD\b": "Kunde",
        r"\bESK\b": "Eskalation",
        r"\bESKA\b": "Eskalation",
        r"\bKK\b": "Kundenkontakt",
        r"\bi\. o\.\b": "in Ordnung",
        r"\bi\.O\b": "in Ordnung",
        r"\bKDG\b": "Kabel Deutschland",  
    }
    
    # Iterate over the dictionary and replace abbreviations in the text
    for abbrev, full_form in abbreviations.items():
        text = re.sub(abbrev, full_form, text)
    
    return text

## failed trial -- I tried to deal with a dictionary instead of a dataframe. It didn't work beacuse operations were more complex and I lost the main power of dealing with newlines instead of \n

# def clean_whitespace(df_explode):
#     df_explode.BearbLog = df_explode.BearbLog.apply(
#         lambda x: x.replace("\r", "")
#         .replace("\\r", "")
#         .replace("\\n", "\n")
#         .replace("\\t\\t", "\n")
#         .replace("\t\t\t","\n")
#     )
#     df_explode.exakte_Problem = df_explode.exakte_Problem.apply(
#         lambda x: x.replace("\r", "")
#         .replace("\\r", "")
#         .replace("\\n", "\n")
#         .replace("\\t\\t", "\n")
#         .replace("\t\t\t","\n")
#     )


# def starter(ticket, ticket_key, ticket_required_keys):
#     df = extract_ticket_data(ticket, ticket_key, ticket_required_keys)

#     df = df.rename(columns= {"desc" : "exakte_Problem",
#                         "produkt" : "dienst_produkt"})

#     df_explode =df.explode(['Datum', 'Benutzer', 'BearbLog'])
#     try:
#         df_explode.BearbLog = df_explode.BearbLog.apply(lambda x: x.replace('\r', '').replace('\\r', '').replace('\\n','\n').replace('\\t\\t','\n').replace("\t\t\t","\n"))
#         df_explode=df_explode.sort_values(by=['Ticketnr','Datum'])
#         df_explode['processed_bearblog'] =df_explode.BearbLog.apply(lambda x:clean_text(extract_fields_with_keys(patterns, x)))
#         df_explode['processed_bearblog'] = df_explode['processed_bearblog'].apply(filter_logs)
#         df_explode['new_processed_bearblog'] = df_explode.processed_bearblog.apply(lambda x: clean_text2(x))
#         df_explode['Benutzer'] =df_explode['Benutzer'].apply(lambda x: x.replace('\n',''))
#     except:
#         print("worklog not found")
#     finally:
#         df_explode.exakte_Problem = df_explode.exakte_Problem.apply(lambda x: x.replace('\r', '').replace('\\r', '').replace('\\n','\n').replace('\\t\\t','\n').replace("\t\t\t","\n"))
#         df_explode['dienst_produkt'] =df_explode['dienst_produkt'].apply(lambda x: x.replace(':',''))
#     return df_explode


def load_fewshots():
    fewshot_df = pd.read_csv("../data/fewshots_without_logs.csv", encoding="Latin-1")
    # this file has the cleaned workglog and the cleaned description for a few tickets
    sum_df = pd.read_csv("../data/summaries_sample.csv", encoding="Latin-1")
    # this file has the summarized workglog and description for the above tickets. It also has the expected summary, which is the merge of worklog and description
    return fewshot_df, sum_df

def load_templates_two_few(fewshot_df, sum_df):
    """
    Here we are providing a general prompt for worklog. Next, we are coupling worklog from few_shot csv and summary from summaries_sample. We are providing this as few shot prompts.

    We are not doing the same for ticket description
    """
    
    # worklog_p = """was im Ticket-Arbeitsprotokoll passiert ist. Beschreiben Sie alle im Arbeitsprotokoll aufgetretenen Aktionsereignisse der Reihe nach. Zeigen Sie keine Kundennamen oder Telefonnummern an. Geben Sie die Ausgabe immer in deutscher Sprache zurück:  """
    
#      worklog_p = """Was ist im Arbeitsprotokoll des Tickets passiert? Beschreiben Sie der Reihe nach die wichtigsten Aktionsereignisse, die im Arbeitsprotokoll aufgetreten sind. Konzentrieren Sie sich besonders auf "Auftrag" und geben Sie nach Möglichkeit das Datum an, wer zugewiesen wurde und um welche Aufgabe es sich handelte.
# Zeigen Sie keine Kundennamen oder Telefonnummern an. Geben Sie die Ergebnisse nur in Aufzählungspunkten ohne Überschriften zurück. Geben Sie die Ausgabe immer auf Deutsch zurück:   """

   #  worklog_p1 = """You will be Given timestamped data of ticket worklog, Give me to the point summary of what happened in the ticket from start till end. You must cover all new orders presented in the ticket which will be highlisted by ' Neur Auftrag' and the order number. your answer must be only in german language.
   # """
#     worklog_p = """Prompt:
    
# Bitte fassen Sie die Arbeitsprotokoll des Tickets erfassten Aktionen und Ereignisse zusammen. Konzentrieren Sie sich auf die detaillierte Beschreibung des erstellten und abgeschlossenen Auftrags, einschließlich Fristen, verantwortlichen Personen und erzielten Ergebnissen. Konzentrieren Sie sich auch auf die Kundenkommunikation, das sind Absätze/Sätze, die mit „Kundeninfo“ beginnen.

# Formatieren Sie die Zusammenfassung wie folgt:
# [Datum]: [Beschreibung].

# [Datum]: [Beschreibung].

# Fassen Sie die Ereignisse in Aufzählungsform und ohne zusätzliche Überschriften zusammen."""
    worklog_p = """Prompt:
Bitte fassen Sie die im Arbeitsprotokoll des Tickets aufgezeichneten Aktionen und Ereignisse zusammen. Konzentrieren Sie sich auf die detaillierte Beschreibung des erstellten und abgeschlossenen Auftrags, einschließlich der Verantwortlichen und der erzielten Ergebnisse. Konzentrieren Sie sich auch auf wichtige Kundenkommunikation und Eskalationen. fasse dich kurz

 Formatieren Sie die Zusammenfassung wie folgt:
 [Datum]: [Beschreibung].

 [Datum]: [Beschreibung].
"""

#     worklog_p = """Prompt:
    
# Fassen Sie die Ereignisse in Aufzählungsform und ohne zusätzliche Überschriften zusammen.

# Formatieren Sie die Zusammenfassung wie folgt:
# [Datum]: [Beschreibung].

# [Datum]: [Beschreibung].

# """
    worklog_system_template = (
        worklog_p
    )
    worklog_template = (
        worklog_system_template
        + """
    - Ticket-Arbeitsprotokoll: {query}
    - Zusammenfassung: 
    """
    )

    worklog_summarization_prompt = PromptTemplate(
        input_variables=["query"], template=worklog_template
    )

    desc_p = """Was ist das in der folgenden Eingabe erwähnte technische Problem? Beschreiben Sie das Problem in der folgenden Eingabe, ohne weitere Informationen hinzuzufügen. Geben Sie die Ausgabe immer in klarer deutscher Sprache zurück:  """

    desc_template = (
        desc_p
        + """
    - input:{query}
    - Problembeschreibung auf deutsch:
    """
    )
    desc_summarization_prompt = PromptTemplate(
        input_variables=["query"], template=desc_template
    )
    return worklog_summarization_prompt, desc_summarization_prompt


# May working code 
# def load_templates(fewshot_df, sum_df):
#     """
#     Here we are providing a general prompt for worklog. Next, we are coupling worklog from few_shot csv and summary from summaries_sample. We are providing this as few shot prompts.

#     We are not doing the same for ticket description
#     """
    
#     # worklog_p = """was im Ticket-Arbeitsprotokoll passiert ist. Beschreiben Sie alle im Arbeitsprotokoll aufgetretenen Aktionsereignisse der Reihe nach. Zeigen Sie keine Kundennamen oder Telefonnummern an. Geben Sie die Ausgabe immer in deutscher Sprache zurück:  """
    
# # second best prompt till now    
# #     worklog_p = """Was ist im Arbeitsprotokoll des Tickets passiert? Beschreiben Sie der Reihe nach die wichtigsten Aktionsereignisse, die im Arbeitsprotokoll aufgetreten sind. Konzentrieren Sie sich besonders auf "Auftrag" und geben Sie nach Möglichkeit das Datum an, wer zugewiesen wurde und um welche Aufgabe es sich handelte.
# # Zeigen Sie keine Kundennamen oder Telefonnummern an. Geben Sie die Ergebnisse nur in Aufzählungspunkten ohne Überschriften zurück. Geben Sie die Ausgabe immer auf Deutsch zurück:   """
   
# # best prompt till now 
#     # worklog_p = """Bitte beschreiben Sie schrittweise die im Arbeitsprotokoll des Tickets aufgezeichneten Aktionen und Ereignisse. Konzentrieren Sie sich auf die detaillierte Darstellung der durchgeführten Aufgaben, einschließlich der jeweiligen Datumsangaben, zuständigen Personen und erzielten Ergebnissen. Include alle relevanten Kundenkontakte und deren Auswirkungen. Fassen Sie die Ereignisse in punktuierter Form zusammen, ohne zusätzliche Überschriften.:    """
#     # worklog_p = """Bitte beschreiben Sie Schritt für Schritt die im Arbeitsprotokoll des Tickets aufgezeichneten Aktionen und Ereignisse. Konzentrieren Sie sich auf die detaillierte Beschreibung der erledigten Aufgaben, einschließlich Daten, verantwortlicher Personen und erzielter Ergebnisse. Fügen Sie nur die relevante Kundenkommunikation ein. Seien Sie prägnant. Fassen Sie die Ereignisse in Aufzählungsform und ohne zusätzliche Überschriften zusammen:   """

#     worklog_p = """Prompt:
# Bitte beschreiben Sie Schritt für Schritt die im Arbeitsprotokoll des Tickets aufgezeichneten Aktionen und Ereignisse. Konzentrieren Sie sich auf die detaillierte Beschreibung der erledigten Aufgaben, einschließlich Daten, verantwortlicher Personen und erzielter Ergebnisse. Fügen Sie nur die relevante Kundenkommunikation ein. Seien Sie prägnant.

# Formatieren Sie die Zusammenfassung wie folgt:

# [Datum]: [Beschreibung der Aktion oder des Ereignisses].

# [Datum]: [Beschreibung der Aktion oder des Ereignisses].

# Beispiel:

# 06.01.2025: Neuer Auftrag "AT0000010923012" wurde angelegt, um die Supportfunktion "LINERESET" durchzuführen.

# 06.01.2025: Der Auftrag wurde erfolgreich abgeschlossen.

# Fassen Sie die Ereignisse in Aufzählungsform und ohne zusätzliche Überschriften zusammen."""

# # the above example is not improving the results 
    

#    #  worklog_p1 = """You will be Given timestamped data of ticket worklog, Give me to the point summary of what happened in the ticket from start till end. You must cover all new orders presented in the ticket which will be highlisted by ' Neur Auftrag' and the order number. your answer must be only in german language.
#    # """

#     worklog_system_template = (
#         worklog_p
#         + f"""      
        
#      - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017183122']['bearblog'].values)}
#      - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017183122']['worklog_summary'].values[0]}

#      - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017173287']['bearblog'].values)}
#      - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017173287']['worklog_summary'].values[0]}
     
#           - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017156983']['bearblog'].values)}
#     - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017156983']['worklog_summary'].values[0]}

#     - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017180051']['bearblog'].values)}
#  - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017180051']['worklog_summary'].values[0]}

#    - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017153499']['bearblog'].values)}
#    - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017153499']['worklog_summary'].values[0]}

# - Ticket-Arbeitsprotokoll: {''.join(fewshot_df[fewshot_df.Ticketnr == 'TA0000017191945']['bearblog'].values)}
# - Zusammenfassung in Stichpunkten in klarer deutscher Sprache: {sum_df[sum_df.Ticketnr == 'TA0000017191945']['worklog_summary'].values[0]}
     
#     """
#     )
#     worklog_template = (
#         worklog_system_template
#         + """
#     - Ticket-Arbeitsprotokoll: {query}
#     - Zusammenfassung in Stichpunkten in klarer deutscher Sprache:
#     """
#     )

#     worklog_summarization_prompt = PromptTemplate(
#         input_variables=["query"], template=worklog_template
#     )

#     desc_p = """Was ist das in der folgenden Eingabe erwähnte technische Problem? Beschreiben Sie das Problem in der folgenden Eingabe, ohne weitere Informationen hinzuzufügen. Geben Sie die Ausgabe immer in klarer deutscher Sprache zurück:  """

#     desc_template = (
#         desc_p
#         + """
#     - input:{query}
#     - Problembeschreibung auf deutsch:
#     """
#     )
#     desc_summarization_prompt = PromptTemplate(
#         input_variables=["query"], template=desc_template
#     )
#     return worklog_summarization_prompt, desc_summarization_prompt


def get_proaktive_desc(text):
    Anschlusskennung_pattern = re.compile(r"Anschlusskennung:\s*(\S+)", re.DOTALL)
    Anschlusskennun_matches = re.findall(Anschlusskennung_pattern, text)
    Anschlusskennun = Anschlusskennun_matches[0] if Anschlusskennun_matches else ""

    Proaktive_pattern = re.compile(
        r"Proaktive Ticketerstellung aufgrund Alarms|Proaktive Ticketerstellung"
    )
    proaktive_matches = re.findall(Proaktive_pattern, text)
    proaktive = proaktive_matches[0] if proaktive_matches else ""

    pattern = re.compile(r"Mitteilung aus NGSA:[^\n]*?\b(Prio\s+\d+\s+Totalausfall)\b")
    totalausfal_matches = re.findall(pattern, text)
    totalausfal = totalausfal_matches[0] if totalausfal_matches else ""
    # totalausfal = totalausfal.replace("\n", "- ")
    totalausfal_text = "Proaktive Ticketerstellung aufgrund Alarms bei dem Regelweg"
    Backup_text = "Proaktive Ticketerstellung aufgrund Alarms bei der BU Verbindung"
    
    return totalausfal_text if totalausfal else Backup_text
    
    # return f"{proaktive} mit der Anschlusskennun {Anschlusskennun} ({totalausfal})"


# def summarize_worklog(df, model):
#     worklog_p = """Sie erhalten zeitgestempelte Daten eines Ticket-Arbeitsprotokolls. Geben Sie eine prägnante und informative Zusammenfassung dessen, was im Ticket von Anfang bis Ende passiert ist. 
# Stellen Sie sicher, dass alle neuen Aufträge, die im Ticket präsentiert werden und durch 'Neuer Auftrag' hervorgehoben sind, sowie die Auftragsnummern, die mit AT000 beginnen, abgedeckt sind.
# Die Zusammenfassung sollte in einem informativen Stil verfasst sein und in der dritten Person (sie/er/es) berichten.
# Ihre Antwort muss ausschließlich in deutscher Sprache erfolgen.
#    """
#     df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M:%S")
#     df["day"] = df["Datum"].dt.strftime("%d-%m-%Y")
#     unique_days = df.day.unique()
#     results = []
#     for day in unique_days:
#         daily_data = df[df["day"] == day]
#         combined_bearblog = ""
#         for _, row in daily_data.iterrows():
#             combined_bearblog += f"{row.Datum}: {row.new_processed_bearblog}\n"
#         prompt = (
#             worklog_p
#             + " worklog: "
#             + combined_bearblog
#             + " Summary steps in german language: "
#         )
#         results.append(model.invoke(prompt).content)
#     sentence_number = 1
#     combined_result = ""
#     for result in results:
#         sentences = result.split("\n")
#         for sentence in sentences:
#             if sentence not in [".", "", " "]:
#                 combined_result += f"{sentence_number}. {sentence.strip()}\n"
#                 sentence_number += 1
#     return combined_result


def is_proaktive(desc, index):
    """Helper function to check if 'Proaktive' is at a specific index in the description."""
    return desc.split(" ")[index] == "Proaktive"

def process_proaktive_desc(desc, produkt):
    """Helper function to process the description and add the product info."""
    desc = desc.replace("ÖProaktive", "Proaktive") if "ÖProaktive" in desc else desc
    return get_proaktive_desc(desc) + f"({produkt})  \n\n"

def clean_whitespace(text):
    text = text.replace("\r", "").replace("\\r", "").replace("\\n", "\n").replace("\\t\\t", "\n").replace("\t\t\t", "\n")
    # Replace more than one space with a single space
    text = re.sub(r' +', ' ', text)
    # Replace more than one consecutive dot with a single dot
    text = re.sub(r'\.{1,}', '.', text)
    return text

def summarize(
    df,
    model,
    worklog_summarization_prompt,
    desc_summarization_prompt,
):
    """Generates a summary of ticket data, including worklog and problem description."""
    worklog = " \n".join(df["processed_bearblog"].values)
    desc = df["exakte_Problem"].values[0]
    produkt = df["dienst_produkt"].values[0]

    if (desc == "No Ticket Description was found!!"):
        worklog_res = model.invoke(
            worklog_summarization_prompt.format(query=worklog)
        )
        results = (
            f"Es wurde keine Beschreibung für Produkt {produkt} gefunden. aber hier ist die Zusammenfassung des Arbeitsprotokolls:\n"
            + worklog_res.content
        )
        return results

    else:
        # Original working implementation
        # if (desc.split(" ")[0] == "Proaktive") | (desc.split(" ")[1] == "Proaktive"):
        #     desc_res = get_proaktive_desc(desc)+ f"({produkt})  \n\n"
        # elif desc.split(" ")[1] == "ÖProaktive":
        #     desc = desc.replace("ÖProaktive", "Proaktive")
        #     desc_res = get_proaktive_desc(desc) + f"({produkt})  \n\n"
        
        if is_proaktive(desc, 0) or is_proaktive(desc, 1):
            desc_res = process_proaktive_desc(desc, produkt)
        elif is_proaktive(desc, 1):
            desc_res = process_proaktive_desc(desc, produkt)

        else:
            desc_res = (
                model.invoke(
                    desc_summarization_prompt.format(query=desc)
                ).content
                + f"({produkt})  \n\n"
            )

        worklog_res = model.invoke(
            worklog_summarization_prompt.format(query=worklog)
        ).content
        results = (
            "Problembeschreibung: "
            + desc_res
            + "\n"
            # + "Zusammenfassung: "
            # + "\n"
            + worklog_res
        )
    return results


# - get cleaned worklog ---------------------------------------------------


# def get_cleaned_worklog(ticket_description, ticket_key, ticket_required_keys):
#     """
#     takes in a new ticket and returns the worklog cleaned
#     """

#     df_explode = starter(ticket_description, ticket_key, ticket_required_keys)
#     new_df, _ = final_processing(df_explode)
#     cleaned_bearblog = new_df["bearblog"].values[0]
#     return cleaned_bearblog


# - extract top worklog events ------------------------------------------------------


# def extract_top_worklog_events(
#     ticket_description, ticket_key, ticket_required_keys, n=0
# ):
#     """
#     This function takes in the cleaned worklog, removes date entries and returns the top n events in the worklog.
#     The function works only if worklog is available in the ticket
#     """
#     try:
#         cleaned_worklog = get_cleaned_worklog(
#             ticket_description, ticket_key, ticket_required_keys
#         )

#         # Split the log into entries based on newlines
#         entries = cleaned_worklog.split("\n")

#         # Use regex to remove dates and timestamps at the beginning of each entry
#         events = [
#             re.sub(r"^\d{2}\.\d{2}\.\d{4}\d{2}:\d{2}:\d{2}: ", "", entry)
#             for entry in entries
#             if entry.strip()
#         ]

#         number_of_events = len(events)

#         # Get the top 'n' events
#         top_n_events = events[:n] if n > 0 and n <= number_of_events else events

#         # Join the events with both a period and a new line after each event
#         return ".\n".join(top_n_events) + "."

#     except:
#         return "Es gibt keine information"


# def batch_clean_summarization_tickets(source_dir="../data/raw_summ_tickets"):
#     """
#     Reads .rep summarization ticets from source directory and appends them to a cleaned  file.
    
#     Args:
#         source_dir (str): Directory containing .rep files
#     """

#     results_all = [] 

#     # Loop over all .rep files in the directory 
#     for filename in os.listdir(source_dir):
#         if filename.endswith(".rep"):
#             file_path = os.path.join(source_dir, filename)

#             # Detect encoding
#             with open(file_path, "rb") as file:
#                 raw_data = file.read()
#                 result = chardet.detect(raw_data)
#                 charenc = result["encoding"]

#             # Read file content
#             with open(file_path, "r", encoding=charenc, errors="replace") as file:
#                 file_content = file.read()

#                 df_explode = starter(file_content, config.ticket_key, config.ticket_required_keys)
#                 new_df, df = final_processing(df_explode)
#                 worklog = new_df.bearblog.values[0]
            
#                 # Split the ticket outside the f-string
#                 ticket_first_line = file_content.split('\n')[1]
#                 results_all.append(f"{ticket_first_line}\n{worklog}\n{'='*50}\n")
#     return results_all


def _read_file(file_path):
    """
    Helper function to read a .rep file and detect its encoding.

    Args:
        file_path (str): Path to the .rep file.

    Returns:
        str: The content of the file.
    """
    # Detect encoding
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        charenc = result["encoding"]

    # Read file content
    with open(file_path, "r", encoding=charenc, errors="replace") as file:
        file_content = file.read()

    return file_content


def _process_file(file_content):
    """
    Helper function to process the content of a .rep file.

    Args:
        file_content (str): The content of the .rep file.

    Returns:
        tuple: A tuple containing the processed DataFrame (`new_df`, `df`) and the file content.
    """
    df_explode = starter(file_content, config.ticket_key, config.ticket_required_keys)
    new_df, df = final_processing(df_explode)
    return new_df, df, file_content


def _read_and_process_file(file_path):
    """
    Wrapper function to read and process a .rep file.

    Args:
        file_path (str): Path to the .rep file.

    Returns:
        tuple: A tuple containing the processed DataFrame (`new_df`, `df`) and the file content.
    """
    file_content = _read_file(file_path)
    return _process_file(file_content)


# def _read_and_process_file(file_path):
#     """
#     Helper function to read and process a .rep file.

#     Args:
#         file_path (str): Path to the .rep file.

#     Returns:
#         tuple: A tuple containing the processed DataFrame (`new_df`, `df`) and the file content.
#     """
#     # Detect encoding
#     with open(file_path, "rb") as file:
#         raw_data = file.read()
#         result = chardet.detect(raw_data)
#         charenc = result["encoding"]

#     # Read file content
#     with open(file_path, "r", encoding=charenc, errors="replace") as file:
#         file_content = file.read()

#         df_explode = starter(file_content, config.ticket_key, config.ticket_required_keys)
#         new_df, df = final_processing(df_explode)
#         return new_df, df, file_content

def _process_files_in_directory(source_dir, process_function):
    """
    Helper function to process all .rep files in a directory.

    Args:
        source_dir (str): Directory containing .rep files.
        process_function (function): Function to process each file.

    Returns:
        list: List of processed results.
    """
    results_all = []

    # Loop over all .rep files in the directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".rep"):
            print(filename)
            file_path = os.path.join(source_dir, filename)
            new_df, df, file_content = _read_and_process_file(file_path)
            result = process_function(new_df, df, file_content)
            results_all.append(result)

    return results_all

def _clean_summarization_tickets_processor(new_df, df, file_content):
    """
    Processes a single file for batch_clean_summarization_tickets.

    Args:
        new_df: Processed DataFrame.
        df: Original DataFrame.
        file_content: Content of the file.

    Returns:
        str: Formatted result for the file.
    """
    # Extract the worklog and format the result
    worklog = new_df.bearblog.values[0]
    ticket_first_line = file_content.split('\n')[1]
    return f"{ticket_first_line}\n{worklog}\n{'='*50}\n"

def _summarization_processor(new_df, df, file_content, model):
    """
    Processes a single file for batch_summarization.

    Args:
        new_df: Processed DataFrame.
        df: Original DataFrame.
        file_content: Content of the file.

    Returns:
        str: Formatted result for the file.
    """
    fewshot_df, sum_df = load_fewshots()
    worklog_summarization_prompt, desc_summarization_prompt = load_templates(fewshot_df, sum_df)
    desc = new_df.exakte_Problem.values[0]
    produkt = new_df["Produkt"].values[0]
    worklog = new_df.bearblog.values[0]
    results = summarize(
        df,
        worklog,
        desc,
        produkt,
        model,
        worklog_summarization_prompt,
        desc_summarization_prompt,
    )
    ticket_first_line = file_content.split('\n')[1]
    return f"{ticket_first_line}\n{results}\n{'='*50}\n"


def batch_clean_summarization(source_dir="../data/raw_summ_tickets"):
    """
    This is a higher-order functions because it passes a function as an argument to another function
    Reads .rep summarization tickets from source directory and appends them to a cleaned file. Note that this function is not taking the arguments. Since _clean_summarization_tickets_processor is passed as an argument, it is not called directly. It will be called inside
    _process_files_in_directory after _read_and_process_file return the needed arguments. 

    Args:
        source_dir (str): Directory containing .rep files

    Returns:
        list: List of processed ticket summaries.
    """
    return _process_files_in_directory(source_dir, _clean_summarization_tickets_processor)

def batch_summarization(source_dir="../data/raw_summ_tickets",model=None):

    """
    Reads .rep summarization tickets from source directory, summarizes them, and appends them to a cleaned file.
    partial is used to pass an argument to a higher order function (batch_summarization, so that it passes it through to the lower order function _summarization_processor)
    Args:
        source_dir (str): Directory containing .rep files

    Returns:
        list: List of summarized ticket results.
    """
    # Bind additional arguments to `summarization_processor` using `partial`
    process_function = partial(
        _summarization_processor,
        model=model,
    )
    return _process_files_in_directory(source_dir, process_function)



