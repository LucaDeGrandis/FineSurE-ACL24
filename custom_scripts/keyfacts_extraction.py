import argparse
import os
import json
from typing import List, Any, Dict
import asyncio
from openai import AsyncOpenAI
import re


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run OpenAI Inference on a given doc')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The input file in JSON format.'
    )
    parser.add_argument(
        '--openai_key',
        type=str,
        required=True,
        help='The secret token for OpenAI.'
    )
    parser.add_argument(
        '--openai_model',
        type=str,
        required=True,
        help='The openai model.'
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        help='The maximum number of retries when the generation of keyfacts fails.',
        default=2,
    )
    parser.add_argument(
        '--key_facts_n',
        type=int,
        help='The maximum number of key facts the model can extract.',
        default=16,
    )
    parser.add_argument(
        '--input_type',
        type=str,
        help='The input type. It can be either "document" or "reference_summary".',
        default="document",
    )
    parser.add_argument(
        '--examples',
        type=str,
        help='The few-shot examples in a txt file.',
        default="document",
    )

    args = parser.parse_args()

    return args


def write_json_file(
    filepath: str,
    input_dict: Dict[str, Any],
    overwrite: bool = False
) -> None:
    """Write a dictionary into a json
    *arguments*
    *filepath* path to save the file into
    *input_dict* dictionary to be saved in the json file
    *overwrite* whether to force overwriting a file.
        Default is false so you don't delete an existing file.
    """
    if not overwrite:
        assert not os.path.exists(filepath)
    with open(filepath, 'w', encoding='utf8') as writer:
        json.dump(input_dict, writer, indent=4, ensure_ascii=False)


def load_txt_file(
    filepath: str,
    join: bool = False,
) -> List[str]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
        for line in lines:
            data.append(line)
    if join:
        data = "".join(data)
    return data


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str = 'a+',
    overwrite: bool = False
) -> None:
    """Write a list into a jsonl
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of json iterable objects
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except Exception:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')


USER_PROMPT_TEMPLATE_SUMMARY = """You will be provided with a summary. Your task is to decompose the summary into a set of "key facts". A "key fact" is a single fact written as briefly and clearly as possible, encompassing at most 2-3 entities.

Here are nine examples of key facts to illustrate the desired level of granularity:
{examples}

Instruction:
First, read the summary carefully.
Second, decompose the summary into (at most {m}) key facts.

Provide your answer in JSON format. The answer should be a dictionary with the key "key facts" containing the key facts as a list:
{{"key facts" ["<key fact 1>", "<key fact 2>", "<key fact 3>", ...]}}

Summary:
{summary}"""


USER_PROMPT_TEMPLATE_DOCUMENT = """You will be provided with a document. Your task is to identify the most important "key facts" of the document. A "key fact" is a single fact written as briefly and clearly as possible, encompassing at most 2-3 entities.

Here are nine examples of key facts to illustrate the desired level of granularity:
{examples}

Instruction:
First, read the document carefully.
Second, decompose the most important information of the document into (at most {m}) key facts.

Provide your answer in JSON format. The answer should be a dictionary with the key "key facts" containing the key facts as a list:
{{"key facts" ["<key fact 1>", "<key fact 2>", "<key fact 3>", ...]}}

Document:
{summary}"""


async def generate_keyfacts(
    text: str,
    examples: str,
    client: AsyncOpenAI,
    model: str,
    prompt_template: str,
    max_retries: int = 1,
    key_facts_n: int = 16,
) -> List[str]:
    """
    Generate key facts from a summary.
    """
    generation_json = None
    token_dicts = []

    if not text:
        print("WARNING: the input text is empty!")
        return generation_json, token_dicts

    retry_count = 0
    while retry_count < max_retries:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(**{
                        "summary": text,
                        "m": key_facts_n,
                        "examples": examples,
                    })
                }
            ]
        )
        token_dicts.append({
            'input': response.usage.prompt_tokens,
            'output': response.usage.completion_tokens,
            'task': 'keyfacts_generation',
        })

        pattern = r"\{[^}]*\}"

        try:
            generation = response.choices[0].message.content
            generation = re.findall(pattern, generation)[0]
            generation_json = json.loads(generation)
        except Exception:
            retry_count += 1
            continue

        return generation_json, token_dicts


def load_json_file(
    filepath: str
) -> List[Any]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    with open(filepath, 'r', encoding='utf8') as reader:
        json_data = json.load(reader)
    return json_data


def load_jsonl_file(
    filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


async def main():
    args = parse_arguments()

    client = AsyncOpenAI(api_key=args.openai_key, max_retries=10)
    MODEL = args.openai_model

    # Generate keyfacts
    data = load_jsonl_file(args.input)
    examples = load_txt_file(args.examples, join=True)
    tasks = []
    assert args.input_type in ['reference_summary', 'document']
    if args.input_type == 'reference_summary':
        USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE_SUMMARY
    else:
        USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE_DOCUMENT
    for line in data:
        if args.input_type == 'reference_summary':
            text_input = line['reference']
        else:
            text_input = line['document']
        tasks.append(generate_keyfacts(text_input, examples, client, MODEL, USER_PROMPT_TEMPLATE, args.max_retries, args.key_facts_n))
    outputs = await asyncio.gather(*tasks)

    keyfacts = []
    for line, output in zip(data, outputs):
        keyfacts.append({
            "source": line["source"],
            "split": line["split"],
            "model": line["model"],
            "doc_id": line['doc_id'],
            "key_facts": output[0]['key facts'],
        })
    out_path = f"{os.path.dirname(args.input)}/keyfacts.json"
    write_jsonl_file(
        out_path,
        keyfacts,
        overwrite=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
