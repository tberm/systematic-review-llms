from ast import literal_eval
import re

import wandb
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
)


REVIEW_LABELS = {
    121733: 'A',
    287708: 'B',
    165805: 'C',
    258698: 'D',
    117787: 'E',
    334317: 'F',
    378562: 'G',
    288055: 'H',
    240084: 'I',
}


def cal_f1(df):
    pos_probs = df.label_probs.apply(lambda p: literal_eval(p)[1])
    curve_points = precision_recall_curve(df.label, pos_probs)
    best_f1 = 0
    for prec, rec, thres in zip(*curve_points):
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def get_wandb_runs_table(project, config_cols=None, summary_cols=None):
    api = wandb.Api()
    runs = api.runs(project)
    return make_df_from_wandb_runs(runs, config_cols, summary_cols)


def make_df_from_wandb_runs(runs, config_cols=None, summary_cols=None):
    if isinstance(config_cols, list):
        config_cols = {col: col for col in config_cols}
    if isinstance(summary_cols, list):
        summary_cols = {col: col for col in summary_cols}

    rows = []
    for run in runs:
        if run.state != 'finished':
            continue
        # getting the created time seems to really slow things down
        #row = {'name': run.name, 'id': run.id, 'created': run.metadata['startedAt']}
        row = {'name': run.name, 'id': run.id}
        run_dicts = [run.summary._json_dict, run.config]
        select_dicts = [summary_cols, config_cols]
        for run_dict, select_dict in zip(run_dicts, select_dicts):
            if select_dict is None:
                for key, val in run_dict.items():
                    if key.startswith('_'):
                        continue
                    if isinstance(val, dict):
                        for subkey, subval in val.items():
                            row[f'{key}/{subkey}'] = subval
                    else:
                        row[key] = val
            else:
                for col, new_name in select_dict.items():
                    if '/' in col:
                        key1, key2 = col.split('/')
                        val = run_dict[key1].get(key2)
                    else:
                        val = run_dict.get(col)

                    row[new_name] = val

        rows.append(row)
    return pd.DataFrame(rows)


def add_pos_rate_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')
    pos_rate = results_df.predicted_label.mean()
    run.summary['pos_rate'] = pos_rate
    run.summary.update()


def add_cal_f1_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')
    run.summary['cal_f1'] = cal_f1(results_df)
    run.summary.update()


def add_neg_recall_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')
    neg_egs = results_df[results_df.label != 1]
    run.summary['neg_recall'] = (neg_egs.predicted_label != 1).mean()
    run.summary.update()


def add_av_prec_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')

    results_df['pos_prob'] = results_df.label_probs.apply(lambda p: literal_eval(p)[1])
    ap = average_precision_score(results_df.label, results_df.pos_prob)
    run.summary['average_precision'] = ap
    run.summary.update()


def add_auroc_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')

    if 'yes_probability' in results_df:
        results_df['pos_prob'] = results_df.yes_probability
    else:
        results_df['pos_prob'] = results_df.label_probs.apply(lambda p: literal_eval(p)[1])

    auroc = roc_auc_score(results_df.label, results_df.pos_prob)
    run.summary['auroc'] = auroc
    run.summary.update()


def add_threshold_scores_to_eval_run(run):
    results_art = [art for art in run.logged_artifacts() if 'Eval' in art.name][0]
    table = results_art.get('Eval results')
    results_df = pd.DataFrame(data=table.data, columns=table.columns).set_index('index')

    # gpt output
    if 'yes_probability' in results_df:
        results_df['pos_prob'] = results_df.yes_probability
    else:
        results_df['pos_prob'] = results_df.label_probs.apply(lambda p: literal_eval(p)[1])

    # tpr: out of all real positives, how many do we get right
    # fpr: out of all real negatives, how many do we get wrong
    fprs, tprs, thresolds = roc_curve(
        results_df.label, results_df.pos_prob, drop_intermediate=True
    )
    for fpr, tpr in zip(fprs, tprs):
        if tpr < 0.95:
            continue
        if 'specificity@95' not in run.summary:
            run.summary['specificity@95'] = 1 - fpr
        if tpr >= 0.98 and 'specificity@98' not in run.summary:
            run.summary['specificity@98'] = 1 - fpr
        if tpr >= 0.99:
            run.summary['specificity@99'] = 1 - fpr
            break

    precs, recs, thresholds = precision_recall_curve(
        results_df.label, results_df.pos_prob, drop_intermediate=True
    )
    revved = (reversed(precs), reversed(recs))
    for prec, rec in zip(*revved):
        if rec < 0.95:
            continue
        if 'precision@95' not in run.summary:
            run.summary['precision@95'] = prec
        if rec >= 0.98 and 'precision@98' not in run.summary:
            run.summary['precision@98'] = prec
        if rec >= 0.99:
            run.summary['precision@99'] = prec
            break

    #if 'precision@98' not in run.summary:
    #    run.summary['precision@98'] = precs[0]
    #if 'precision@99' not in run.summary:
    #    run.summary['precision@99'] = precs[0]

    run.summary.update()


def get_results_df_for_run(run):
    results_arts = [art for art in run.logged_artifacts() if 'eval' in art.name.lower()]
    if len(results_arts) > 1:
        art = sorted(results_arts, key=lambda art: int(art.name[-1]))[-1]
        print('WARNING: run contains several eval tables. Using', art.name)
    else:
        art = results_arts[0]

    table = art.get('Eval results')
    return pd.DataFrame(data=table.data, columns=table.columns).set_index('index')

def get_model_from_run(run):
    arts = [
        art for art in run.logged_artifacts()
        if 'model' in art.name
    ]
    return sorted(arts, key=lambda art: int(art.version[1:]))[-1]


def get_review_from_eval_set(eval_set):
    match = re.search(r'/review_(\d{6})/', eval_set)
    return int(match.groups()[0])



def get_review_from_model_name(model_name, no_version=True):
    runs_to_reviews = {
        'bluebell': 117787,
        'daffodil': 121733,
        'hyacinth': 258698,
        'edelweiss': 287708,
        'lily': 334317,
        'iris': 378562,
        'poppy': 165805,
    }
    if no_version:
        match = re.search(r'model-(\w+)(-(cc|sm))?', model_name)
    else:
        match = re.search(r'model-(\w+)(-(cc|sm))?:v\d', model_name)
    run_name = match.groups()[0]
    return runs_to_reviews[run_name]


def init_mpl():
    import matplotlib as mpl
    from cycler import cycler 

    #mpl.rcParams['axes.prop_cycle'] = cycler(color=['#065143', '#93B7BE', '#EE8434'])
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['font.size'] = 11
