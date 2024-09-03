from pathlib import Path
import torch
import yaml
from run_common import load_reviews


def format_without_paper(title, criteria):
    here_dir = Path(__file__).parent
    with (here_dir / 'prompt_parts.yaml').open(encoding='utf-8') as parts_file:
        prompt_parts = yaml.load(parts_file, Loader=yaml.SafeLoader)

    prompt = prompt_parts['instruction'].format(REVIEW_TITLE=title, CRITERIA=criteria)

    return prompt


def get_criteria_embedding(
        model, tokenizer, review_id, layer, device='cpu',
        as_instruction=False
):
    review = load_reviews().loc[review_id]
    if as_instruction:
        prompt = format_without_paper(review.title, review.criteria)
    else:
        prompt = review.criteria

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    mean_over_pos = outputs.hidden_states[layer][0].mean(dim=0)
    return mean_over_pos.flatten().detach()

