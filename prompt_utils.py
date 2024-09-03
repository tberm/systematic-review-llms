from pathlib import Path
import functools
import yaml
import pandas as pd
from global_config import CONFIG


LABELS = [
    "No",
    "Yes",
    "Pass",
]

# indexes of training set items to use as few shot examples
# for n-shot learning, we use the first n from this list
# they are selected such that for any n we have a reasonable balance of labels and the
# papers are not too similar to each other
# FEW_SHOT_EXAMPLES = [2809, 1297, 1335, 3127, 839, 2656, 569, 3299, 2392, 3497]


def format_input(title, abstract, label=None):
    """
    Format title/abstract input, optionally adding the label if it's a few-shot learning
    example
    """
    out = f"Title: {title}\n\nAbstract: {abstract}\n\nDecision:"

    if label is not None:
        assert label in (0,1)
        if CONFIG.include_space_in_labels:
            out += " " + LABELS[int(label)]
        else:
            out += LABELS[int(label)]

    return out


def assemble_prompt(
    review_title, criteria, input_title, input_abs, label=None,
    num_examples=0
):
    """
    Assemble and tokenize prompt for zero-shot or few-shot task using Llama-3 prompt
    format.
    """
    # some abstracts are really just full essays; truncate so we don't run out of memory
    if len(input_abs) > CONFIG.max_abstract_chars:
        input_abs = input_abs[:CONFIG.max_abstract_chars]
    if len(input_title) > CONFIG.max_title_chars:
        input_title = input_title[:CONFIG.max_title_chars]

    here_dir = Path(__file__).parent
    with (here_dir / 'prompt_parts.yaml').open(encoding='utf-8') as parts_file:
        prompt_parts = yaml.load(parts_file, Loader=yaml.SafeLoader)

    #prompt = prompt_parts['short_sys_msg']
    prompt = ''
    if CONFIG.run_config.get('allow_pass'):
        instr = prompt_parts['instruction_allow_pass'] 
    else:
        instr = prompt_parts['instruction'] 
    prompt += '\n\n' + instr.format(
        REVIEW_TITLE=review_title,
        CRITERIA=criteria,
    )

    paper_tpl = prompt_parts['paper']
    if num_examples > 0:
        examples = get_examples(num_examples)
        ex_string = '\n\n'.join([
            paper_tpl.format(**ex) + " " + LABELS[int(ex['label'])]
            for ex in examples
        ])

        prompt += '\n\n' + prompt_parts['examples'].format(EXAMPLES=ex_string)
        prompt += '\n\n' + prompt_parts['reinstruct']

    prompt += '\n\n' + paper_tpl.format(title=input_title, abstract=input_abs)

    return prompt


def assemble_prompt_chat_model(
    tokenizer, review, input_title, input_abs,
    num_examples=0, prompt_format='chat_history', model_class='llama',
    return_messages=False, remind=False, reverse=False, short_criteria=False,
    incl_title=True,
):
    """
    Assemble and tokenize prompt for zero-shot or few-shot task using Llama-3 prompt
    format.
    """
    here_dir = Path(__file__).parent
    with (here_dir / 'prompt_parts.yaml').open(encoding='utf-8') as parts_file:
        prompt_parts = yaml.load(parts_file, Loader=yaml.SafeLoader)

    if remind and reverse:
        raise ValueError("`remind` and `reverse` are incompatible")

    if model_class == 'mistral' and prompt_format == 'system_message':
        raise ValueError("Cannot use `system_message` prompt format:"
                         " mistral doesn't have system role")

    # some abstracts are really just full essays; truncate so we don't run out of memory
    if len(input_abs) > CONFIG.max_abstract_chars:
        input_abs = input_abs[:CONFIG.max_abstract_chars]
    if len(input_title) > CONFIG.max_title_chars:
        input_title = input_title[:CONFIG.max_title_chars]

    if short_criteria:
        criteria = review[f'{model_class}_criteria_summary']
    else:
        criteria = review['criteria']

    if CONFIG.run_config.get('allow_pass'):
        instruction = prompt_parts['instruction_allow_pass']
    if not incl_title:
        instruction = prompt_parts['instruction_no_title']
    else:
        instruction = prompt_parts['instruction']

    if prompt_format == 'user_message':
        #sys_msg = prompt_parts['short_sys_msg']
        sys_msg = ''

        if reverse:
            user_msg = prompt_parts['instruction_no_criteria']
        else:
            user_msg = instruction.format(
                REVIEW_TITLE=review['title'],
                CRITERIA=criteria,
            )

        paper_tpl = prompt_parts['paper']
        if num_examples > 0:
            examples = get_examples(num_examples)
            ex_string = '\n\n'.join([
                paper_tpl.format(**ex) + " " + LABELS[int(ex['label'])]
                for ex in examples
            ])

            user_msg += '\n\n' + prompt_parts['examples'].format(EXAMPLES=ex_string)

            if not (remind or reverse):
                user_msg += '\n\n' + prompt_parts['reinstruct']

        if remind:
            user_msg += '\n\n' + prompt_parts['reminder'].format(
                SHORT_CRITERIA=review[f'{model_class}_criteria_summary']
            )

        if reverse:
            user_msg += '\n\n' + prompt_parts['criteria_no_instructions'].format(
                REVIEW_TITLE=review['title'],
                CRITERIA=criteria,
            ) + '\n\n' + "The details of the paper to consider are below."

        user_msg += '\n\n' + paper_tpl.format(title=input_title, abstract=input_abs)

        if model_class == 'mistral':
            messages = [
                # mistral doesn't accept a system role
                {'role': 'user', 'content': sys_msg + '\n\n' + user_msg},
            ]
        else:
            messages = [
                #{'role': 'system', 'content': sys_msg},
                {'role': 'user', 'content': user_msg},
            ]

        if return_messages:
            return messages

        return tokenizer.apply_chat_template(
            messages, return_tensors='pt', padding=True, add_generation_prompt=True,
            tokenize=False,
        )

    elif prompt_format in ('system_message', 'biomed_llama'):
        #sys_msg = prompt_parts['short_sys_msg']
        sys_msg = ''

        if reverse:
            sys_msg = prompt_parts['instruction_no_criteria']
        else:
            #sys_msg += '\n\n' + instruction.format(
            sys_msg += instruction.format(
                REVIEW_TITLE=review['title'],
                CRITERIA=criteria,
            )

        paper_tpl = prompt_parts['paper']
        if num_examples > 0:
            examples = get_examples(num_examples)
            ex_string = '\n\n'.join([
                paper_tpl.format(**ex) + " " + LABELS[int(ex['label'])]
                for ex in examples
            ])

            sys_msg += '\n\n' + prompt_parts['examples'].format(EXAMPLES=ex_string)

        if reverse:
            sys_msg += '\n\n' + prompt_parts['criteria_no_instructions'].format(
                REVIEW_TITLE=review['title'],
                CRITERIA=criteria,
            )

        if remind:
            sys_msg += '\n\n' + prompt_parts['reminder'].format(
                SHORT_CRITERIA=review[f'mistral_criteria_summary']
            )

        user_msg = paper_tpl.format(title=input_title, abstract=input_abs)

        if prompt_format == 'biomed_llama':
            prompt_tpl = prompt_parts['biomed_llama_scaffold'] + '\n'
            prompt = prompt_tpl.format(
                INSTRUCTION=sys_msg,
                INPUT=user_msg,
            )
            if return_messages:
                raise ValueError('`biomed_llama` format does not use messages')
            return prompt


        messages = [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg},
        ]

        if return_messages:
            return messages

        return tokenizer.apply_chat_template(
            messages, return_tensors='pt', padding=True, add_generation_prompt=True,
            tokenize=False,
        )

    if remind:
        raise ValueError("Can't have reminder in chat history format")

    sys_msg = instruction.format(
        REVIEW_TITLE=review['title'],
        CRITERIA=criteria,
    )

    paper_tpl = prompt_parts['paper']
    user_msg = paper_tpl.format(title=input_title, abstract=input_abs)

    messages = [{
        'role': 'user' if model_class == 'mistral' else 'system',
        'content': sys_msg
    }]
    for ex in get_examples(num_examples):
        sep = " " if CONFIG.include_space_in_labels else ""
        messages += [
            {'role': 'user', 'content': paper_tpl.format(**ex)},
            {'role': 'assistant', 'content': sep + LABELS[int(ex['label'])]},
        ]

    messages.append({'role': 'user', 'content': user_msg})

    if model_class == 'mistral':
        messages[0] = {
            'role': 'user',
            'content': messages[0]['content'] + '\n\n' + messages.pop(1)['content'],
        }

    if return_messages:
        return messages
 
    return tokenizer.apply_chat_template(
        messages, return_tensors='pt', padding=True, add_generation_prompt=True,
        tokenize=False,
    )


@functools.lru_cache
def get_examples(number):
    if number == 0:
        return []
    here_dir = Path(__file__).parent
    ex_set_path = CONFIG.run_config['examples_set']
    if ex_set_path[0] != '/':
        ex_set_path = here_dir / ex_set_path
    ex_df = pd.read_csv(ex_set_path, index_col=0)
    indexes = CONFIG.run_config['example_ids']
    rows = ex_df.loc[indexes]
    return [
        {'title': row.title, 'abstract': row.abstract, 'label': row.label}
        for _, row in rows.iterrows()
    ]
