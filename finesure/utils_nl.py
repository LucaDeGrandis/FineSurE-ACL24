import nltk
import json


nltk.download("punkt_tab")


MIN_SENT_LEN = 5


async def get_response(client, messages, model, temperature=0.0):

    ''' A function to get the response from GPT-series
    Args:
        client: openai client
        prompt: input prompt
        model: openai model name
    Return:
        text_response: the output from LLMs
    '''

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    text_response = response.choices[0].message.content

    return text_response


KEYFACT_ALIGN_SYSTEM_TEMPLATE = \
    """You are a summary evaluator. Your job is to associate summary sentences with source sentences.
The inputs are a list of summary sentences and the document source. The expected output is a list of summary sentences associated with source sentences.


Do the following:
1. For each summary sentence, identify the corresponding source sentences.
2. If the summary sentence is not associated with any source sentences, skip it.


Answer in JSON format. Here is an example of the output format:
[{"summary_sentence": "summary sentence 1", "source_sentences": ["source sentence 1", "source sentence"]},{"summary_sentence": "summary sentence 2", "source_sentences": ["source sentence", ...]}, ...]"""

KEYFACT_ALIGN_USER_TEMPLATE = \
    """Here is the document source:
[document_start]
{document}
[document_end]


Here is the list of summary sentences:
[sentences_start]
{sentences}
[sentences_end]


Now associate summary sentences with source sentences."""


def get_keyfact_alighment_prompt(summary_sentences, source_document):
    sentences = summary_sentences
    sentences = list(filter(lambda x: len(nltk.word_tokenize(x)) >= MIN_SENT_LEN, sentences))
    sentences_flat = ""
    for sent in sentences:
        sentences_flat += f"- {sent}\n"
    sentences_flat = sentences_flat.strip()
    messages = [
        {"role": "system", "content": KEYFACT_ALIGN_SYSTEM_TEMPLATE},
        {"role": "user", "content": KEYFACT_ALIGN_USER_TEMPLATE.format(**{
            "document": source_document,
            "sentences": sentences_flat
        })}
    ]

    return messages


def parsing_llm_keyfact_alighment_output(output, document):
    document = document.lower()
    try:
        while not output.startswith("{"):
            if not output:
                break
            output = output[1:]
        while not output.endswith("}"):
            if not output:
                break
            output = output[:-1]
        output_json = json.loads(output)

        document_sentences = [x.lower() for x in nltk.sent_tokenize(document)]
        document_sentences = list(filter(lambda x: len(nltk.word_tokenize(x)) >= MIN_SENT_LEN, document_sentences))

        pred_labels = []
        matched_lines = set()
        for out in output_json:
            source_sentences = [x.lower().strip() for x in out["source_sentences"]]
            found_sentences = list(filter(lambda x: any(x in y for y in document_sentences), source_sentences))

            if found_sentences:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

            for sent in document_sentences:
                if any(x in sent for x in found_sentences):
                    matched_lines.add(sent)

        return pred_labels, list(matched_lines)

    except Exception as e:
        print(e)
        return [], []


def compute_conciseness_percentage_score(pred_sentence_line_numbers, num_sentences):
    if num_sentences == 0:
        return 0.0
    conciseness = len(pred_sentence_line_numbers) / num_sentences
    return conciseness


FACT_CHECK_SYSTEM_TEMPLATE = \
    """You are a summary evaluator. Your job is to associate summary sentences with source sentences and evaluate whether the summary sentence is entailed in the identified source sentences.
The inputs are a list of summary sentences and the document source. The expected output is a list of summary sentences associated with source sentences.


Do the following:
1. For each summary sentence, identify the corresponding source sentences.
2. If the summary sentence is not associated with any source sentences, use an empty list.
3. Decide whether the summary sentence is entailed in the identified source sentences. Use "ENTAILED" or "NOT_ENTAILED" to answer.


Answer in JSON format. Here is an example of the output format:
[{"summary_sentence": "summary sentence 1", "source_sentences": ["source sentence 1", "source sentence"], "entailment": "ENTAILED"}, {"summary_sentence": "summary sentence 2", "source_sentences": ["source sentence", ...], "entailment": "NOT_ENTAILED"}, ...]"""

FACT_CHECK_USER_TEMPLATE = KEYFACT_ALIGN_USER_TEMPLATE


def get_fact_checking_prompt(summary_sentences, document):
    sentences_flat = ""
    for line in summary_sentences:
        sentences_flat += f"- {line}\n"
    sentences_flat = sentences_flat.strip()

    messages = [
        {"role": "system", "content": FACT_CHECK_SYSTEM_TEMPLATE},
        {"role": "user", "content": FACT_CHECK_USER_TEMPLATE.format(**{
            "document": document,
            "sentences": sentences_flat
        })}
    ]

    return messages


def parsing_llm_fact_checking_output(output):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return:
        pred_labels: the binary label for each sentence (0: no factuality error, 1: factuality error)
        pred_types: the error type of each sentence
    '''

    try:
        while not output.startswith("{"):
            if not output:
                break
            output = output[1:]
        while not output.endswith("}"):
            if not output:
                break
            output = output[:-1]
        output_json = json.loads(output)

        pred_labels, pred_types = [], []
        for out in output_json:
            if out["entailment"] == "ENTAILED":
                pred_labels.append(0)
                pred_types.append("no error")
            else:
                pred_labels.append(1)
                pred_types.append("error")

            return pred_labels, pred_types

    except Exception as e:
        print('parsing error:', e)
        return [], []
