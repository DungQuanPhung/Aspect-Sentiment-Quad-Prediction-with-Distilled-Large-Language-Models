import json
import os
import re
import argparse
import warnings
from collections import defaultdict

# Scoring only needs the data folder; importing `src` would pull in every LLM SDK.
data_path = os.path.join(os.getcwd(), "data")
warnings.filterwarnings("ignore")


def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def read_agents_out(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def read_multiple_json(inference_filename):
    """Concatenates `<prefix>_1.json`, `<prefix>_2.json`, ... written by resumed runs."""
    processed_dir = os.path.join(data_path, "processed")
    pattern = re.compile(rf"^{re.escape(inference_filename)}_(\d+)\.json$")
    matches = sorted((int(m.group(1)), f) for f in os.listdir(processed_dir) if (m := pattern.match(f)))
    if not matches:
        raise FileNotFoundError(f"No prediction files '{inference_filename}_<n>.json' in {processed_dir}")
    data = defaultdict(list)
    for _, json_file in matches:
        with open(os.path.join(processed_dir, json_file), 'r', encoding='utf-8') as f:
            for k, v in json.load(f).items():
                data[k].extend(v)
    return data


def read_timing(inference_filename):
    """Sums the `_timing.json` files of all (possibly resumed) runs for this prefix."""
    timing_dir = os.path.join(data_path, "processed", "timing")
    if not os.path.isdir(timing_dir):
        return None
    pattern = re.compile(rf"^{re.escape(inference_filename)}_(\d+)_timing\.json$")
    total = {"n_instances": 0, "total_sec": 0.0, "n_api_calls": 0, "train_sec": 0.0}
    found = False
    for f in os.listdir(timing_dir):
        if pattern.match(f):
            found = True
            with open(os.path.join(timing_dir, f), 'r', encoding='utf-8') as fh:
                run = json.load(fh)
            for key in total:
                total[key] += run.get(key) or 0
    if not found:
        return None
    total["sec_per_instance"] = total["total_sec"] / total["n_instances"] if total["n_instances"] else 0
    return total


def normalize_quads(quadruples):
    """Case/whitespace-insensitive set of quads, so 'Phòng' and 'phòng' count as the same term."""
    try:
        return set(tuple(str(e).strip().lower() for e in q) for q in quadruples)
    except TypeError:
        warnings.warn("Parsing Error")
        return set()


def calculate_tp_fp_fn_asqp(gold_quadruples, pred_quadruples):
    gold_set = normalize_quads(gold_quadruples)
    pred_set = normalize_quads(pred_quadruples)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def calculate_tp_fp_fn_aste(gold_quadruples, pred_quadruples):
    gold_set = {(a, p, o) for a, c, p, o in normalize_quads(gold_quadruples)}
    pred_set = {(q[0], q[2], q[3]) for q in normalize_quads(pred_quadruples) if len(q) == 4}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def calculate_tp_fp_fn_acsa(gold_quadruples, pred_quadruples):
    gold_set = {(c, p) for a, c, p, o in normalize_quads(gold_quadruples)}
    pred_set = {(q[1], q[2]) for q in normalize_quads(pred_quadruples) if len(q) == 4}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def string_parsing(quadruples_):
    quad_list = list()
    if quadruples_ == "[]":
        return quad_list
    quadruples_ = quadruples_.strip()
    quadruples_ = quadruples_.replace('\n', " ")
    # quadruples_ = quadruples_.replace('  ', " ")
    quadruples_ = ' '.join(quadruples_.split())
    quads = quadruples_.split("], [")
    quads = [quad.replace("[[", "") for quad in quads]
    quads = [quad.replace("]]", "") for quad in quads]
    for quad in quads:
        elements = list()
        quad = quad.replace('"', "'")

        elements = quad.split("', '")
        # elements = [element.split("', \"") for element in elements]
        # elements = [element.split("\", '") for element in elements]
        # elements = [element.split("\", \"") for element in elements]
        # elements = [element.strip('"') for element in elements]
        elements = [element.strip("'") for element in elements]
        elements = [element.strip('\\') for element in elements]
        quad_list.append(elements)
        # if len(elements) != 4:
        #     print("fefe")
    return quad_list


def calculate_fscore(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, fscore


def main(gold_file, pred_file=None, pred_prefix=None):
    gold_sentences = read_sentences(gold_file)
    if pred_file:
        data = read_agents_out(pred_file)
    elif pred_prefix:
        data = read_multiple_json(pred_prefix)
    else:
        raise ValueError("Either pred_prefix or pred_file must be specified")
    # Extract only the amount of labelled data
    gold_sentences = gold_sentences[:len(data[list(data.keys())[0]])]
    # print("Amount of sentences: ", len(data[list(data.keys())[0]]))

    scoring = dict()
    agent_aggregate = defaultdict(list)
    for k, v in data.items():
        gibrish_count = 0
        total_tp_asqp = total_fp_asqp = total_fn_asqp = 0
        total_tp_aste = total_fp_aste = total_fn_aste = 0
        total_tp_acsa = total_fp_acsa = total_fn_acsa = 0
        counts = 0
        for gold_sentence, pred_sentence in zip(gold_sentences, v):
            try:
                try:
                    if "agents" in (pred_prefix or pred_file):
                        quads_pred = string_parsing(pred_sentence[0])  # ast.literal_eval(pred_sentence[0])
                    else:
                        quads_pred = string_parsing(pred_sentence)  # ast.literal_eval(pred_sentence)
                except TypeError:
                    quads_pred = string_parsing(pred_sentence)  # ast.literal_eval(pred_sentence)
            except SyntaxError:
                # print(pred_sentence)
                quads_pred = list()
                gibrish_count += 1
                if "agents" in (pred_prefix or pred_file):
                    warnings.warn(pred_sentence[0])
                else:
                    warnings.warn(pred_sentence)
            for q in quads_pred:
                if len(q) != 4:
                    gibrish_count += 1
                    if "agents" in (pred_prefix or pred_file):
                        warnings.warn(pred_sentence[0])
                    else:
                        warnings.warn(pred_sentence)
            if "agents" in (pred_prefix or pred_file):
                agent_aggregate[k].append(quads_pred)
            # print(quads_pred)
            # print(gold_sentence['quadruples'])
            # print()
            tp, fp, fn = calculate_tp_fp_fn_asqp(gold_sentence['quadruples'], quads_pred)
            total_tp_asqp += tp
            total_fp_asqp += fp
            total_fn_asqp += fn

            tp, fp, fn = calculate_tp_fp_fn_aste(gold_sentence['quadruples'], quads_pred)
            total_tp_aste += tp
            total_fp_aste += fp
            total_fn_aste += fn

            tp, fp, fn = calculate_tp_fp_fn_acsa(gold_sentence['quadruples'], quads_pred)
            total_tp_acsa += tp
            total_fp_acsa += fp
            total_fn_acsa += fn

            # counts += 1
            # if counts == 41:
            #     break
            # print(f"Sentence: {gold_sentence['sentence']}")
            # sentence_precision, sentence_recall, sentence_fscore = calculate_fscore(tp, fp, fn)
            # print(f"  Precision: {sentence_precision:.4f}")
            # print(f"  Recall: {sentence_recall:.4f}")
            # print(f"  F-score: {sentence_fscore:.4f}")
            # print()

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_asqp, total_fp_asqp,
                                                                             total_fn_asqp)
        scoring[k] = {"ASQP_precision": overall_precision, "ASQP_recall": overall_recall, "ASQP_fscore": overall_fscore,
                      "gibberish_outputs": gibrish_count}

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_aste, total_fp_aste,
                                                                             total_fn_aste)
        scoring[k].update(
            {"ASTE_precision": overall_precision, "ASTE_recall": overall_recall, "ASTE_fscore": overall_fscore})

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_acsa, total_fp_acsa,
                                                                             total_fn_acsa)
        scoring[k].update(
            {"ACSA_precision": overall_precision, "ACSA_recall": overall_recall, "ACSA_fscore": overall_fscore})

    sorted_data = dict(sorted(scoring.items(), key=lambda item: item[1]['ASQP_fscore']))
    for k, v in sorted_data.items():
        print(k)
        print("Overall Scores for ASQP (quad-F1):")
        print(f"  Precision: {v['ASQP_precision']:.4f}")
        print(f"  Recall: {v['ASQP_recall']:.4f}")
        print(f"  F-score: {v['ASQP_fscore']:.4f}")
        print(f"  Unparsable outputs: {v['gibberish_outputs']}")
        print()
    timing = read_timing(pred_prefix) if pred_prefix else None
    if timing:
        print(f"Inference time: {timing['total_sec']:.1f}s for {timing['n_instances']} instances "
              f"({timing['sec_per_instance']:.2f}s/instance, {timing['n_api_calls']} API calls)\n")
        if timing["train_sec"]:
            print(f"Training time: {timing['train_sec']:.0f}s ({timing['train_sec'] / 3600:.2f} h)\n")
    results = {"gold_file": gold_file, "predictions": pred_prefix or pred_file,
               "n_evaluated": len(gold_sentences), "timing": timing, "scores": sorted_data}
    scores_dir = os.path.join(data_path, "processed", "scores")
    os.makedirs(scores_dir, exist_ok=True)
    scores_name = pred_prefix or os.path.splitext(os.path.basename(pred_file))[0]
    with open(os.path.join(scores_dir, scores_name + "_scores.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return results
    # if "agents" in (pred_prefix or pred_file):
    #     print("Agent Aggregate")
    #     aggregate = list()
    #     for i in range(len(agent_aggregate[list(agent_aggregate.keys())[0]])):
    #         agent_res = list()
    #         for agent in agent_aggregate.keys():
    #             agent_res.extend(agent_aggregate[agent][i])
    #         aggregate.append(agent_res)
    #     total_tp_asqp = total_fp_asqp = total_fn_asqp = 0
    #     total_tp_aste = total_fp_aste = total_fn_aste = 0
    #     total_tp_acsa = total_fp_acsa = total_fn_acsa = 0
    #     for gold_sentence, pred_sentence in zip(gold_sentences, aggregate):
    #         tp, fp, fn = calculate_tp_fp_fn_asqp(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_asqp += tp
    #         total_fp_asqp += fp
    #         total_fn_asqp += fn
    #         tp, fp, fn = calculate_tp_fp_fn_aste(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_aste += tp
    #         total_fp_aste += fp
    #         total_fn_aste += fn
    #         tp, fp, fn = calculate_tp_fp_fn_acsa(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_acsa += tp
    #         total_fp_acsa += fp
    #         total_fn_acsa += fn

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_asqp, total_fp_asqp,
    #                                                                          total_fn_asqp)
    #     print("Overall Scores for ASQP:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_aste, total_fp_aste,
    #                                                                          total_fn_aste)

    #     print("Overall Scores for ASTE:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_acsa, total_fp_acsa,
    #                                                                          total_fn_acsa)
    #     print("Overall Scores for ACSA:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quad-F1 (ASQP), ASTE and ACSA scores for saved predictions.")
    parser.add_argument("--dataset", default="hotels_vn", help="use e.g. hotels_vn_dev to score a --split dev run")
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--tasks", nargs="+",
                        default=["agents", "icl_0", "icl_2", "icl_10", "icl_20", "icl_50", "icl_100"])
    args = parser.parse_args()

    gold_file = os.path.join(data_path, "processed", args.dataset + "_ground_truth.json")
    for task in args.tasks:
        pred_prefix = f"{re.sub(r'[/:]', '_', args.model)}_{task}_{args.dataset}"
        print(f"___ {pred_prefix} ___")
        try:
            main(gold_file, pred_prefix=pred_prefix)
        except FileNotFoundError as e:
            print(e, "\n")