"""I index frozen local trajectories for later review, without judging their content.

TOY LAB RESEARCH ONLY. I read a JSONL snapshot and write separate derived files.
I never load a model, contact a provider, quote response text, or alter the input.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ARM_FIELDS = ("arm_id", "stage", "variant", "kind", "split", "layer", "alpha", "mask", "max_new_tokens")
INDEX_FIELDS = ("generation_line_1based", "prompt_id", *ARM_FIELDS, "final_started", "final_ended",
                "censored", "raw_output_present", "cot_present", "final_text_present", "n_gen",
                "stop_reason", "provisional_success_recorded", "canned_refusal_recorded")
OUTPUT_NAMES = ("arm-counts.csv", "prompt-arm-index.csv", "prompt-arm-matrix.csv", "coverage.json", "README.md")


def scalar(value):
    """I retain identifiers and metadata; nested structures are not index labels."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("An index identifier or metadata value has an unexpected nested type")


def boolean(value):
    return value if isinstance(value, bool) else None


def read_snapshot(path, allow_partial_tail=False):
    raw = Path(path).read_bytes()
    lines = raw.splitlines()
    records, warnings = [], []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            warnings.append({"line":line_number, "issue":"blank_line"})
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if allow_partial_tail and line_number == len(lines) and not raw.endswith(b"\n"):
                warnings.append({"line":line_number, "issue":"incomplete_final_line_excluded", "bytes":len(line)})
                continue
            raise ValueError(f"Invalid JSON at generation line {line_number}; no index written") from exc
        if not isinstance(row, dict) or "prompt_id" not in row:
            raise ValueError(f"Generation line {line_number} lacks a record object/prompt_id")
        record = {"generation_line_1based":line_number, "prompt_id":scalar(row["prompt_id"])}
        for field in ARM_FIELDS:
            value = row.get(field)
            if field == "arm_id":
                value = value or row.get("arm") or row.get("name") or "unspecified"
            if field == "max_new_tokens" and value is None:
                value = row.get("max_tokens")
            record[field] = scalar(value)
        record.update(
            final_started=boolean(row.get("final_started")),
            final_ended=boolean(row.get("final_ended")), censored=boolean(row.get("censored")),
            raw_output_present=bool(row.get("raw_output", row.get("output", ""))),
            cot_present=bool(row.get("cot", "")), final_text_present=bool(row.get("final", "")),
            n_gen=scalar(row.get("n_gen")), stop_reason=scalar(row.get("stop_reason")),
            provisional_success_recorded=boolean(row.get("provisional_success")),
            canned_refusal_recorded=boolean(row.get("canned_refusal")))
        records.append(record)
    return records, warnings, {"sha256":hashlib.sha256(raw).hexdigest(), "bytes":len(raw),
                               "physical_lines":len(lines), "ends_with_newline":raw.endswith(b"\n")}


def key_string(values):
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_index(generations, out, allow_partial_tail=False):
    source, out = Path(generations).resolve(), Path(out).resolve()
    if source == out or out == source.parent:
        raise ValueError("I require a separate output directory for the derived index")
    if out.exists() and any(out.iterdir()):
        raise ValueError("The index output directory must be new or empty")
    rows, warnings, identity = read_snapshot(source, allow_partial_tail)
    groups = defaultdict(list)
    prompt_groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in ARM_FIELDS)].append(row)
        # Kind prevents an Alpaca ID from being paired with a harmful-prompt ID.
        # Split remains part of identity so selection and holdout never mix.
        prompt_groups[key_string([row["prompt_id"],row["kind"],row["split"]])].append(row)
    arm_rows, arm_columns = [], []
    for number, (key, group) in enumerate(groups.items(), start=1):
        values = dict(zip(ARM_FIELDS,key))
        column = f"arm_{number:03d}"
        arm_columns.append(column)
        values.update(matrix_column=column, rows=len(group), unique_prompts=len({key_string(r['prompt_id']) for r in group}),
            final_started=sum(r["final_started"] is True for r in group),
            final_ended=sum(r["final_ended"] is True for r in group),
            final_ended_unknown=sum(r["final_ended"] is None for r in group),
            censored=sum(r["censored"] is True for r in group),
            censoring_unknown=sum(r["censored"] is None for r in group),
            raw_output_present=sum(r["raw_output_present"] for r in group),
            cot_present=sum(r["cot_present"] for r in group),
            final_text_present=sum(r["final_text_present"] for r in group),
            generation_tokens=sum(r["n_gen"] for r in group if isinstance(r["n_gen"],(int,float)) and not isinstance(r["n_gen"],bool)))
        arm_rows.append(values)
    column_map = {key:column for key,column in zip(groups,arm_columns)}
    matrix, duplicate_cells = [], []
    paired_prompt_count = 0
    for prompt_key, group in prompt_groups.items():
        prompt_id, kind, split = json.loads(prompt_key)
        cell_lines = defaultdict(list)
        for row in group:
            column = column_map[tuple(row[f] for f in ARM_FIELDS)]
            cell_lines[column].append(row["generation_line_1based"])
        paired_prompt_count += len(cell_lines) > 1
        matrix.append({"prompt_id":prompt_id,"kind":kind,"split":split,
                       "available_arm_columns":len(cell_lines),
                       **{column:";".join(map(str,cell_lines[column])) for column in arm_columns}})
        for column, line_numbers in cell_lines.items():
            if len(line_numbers) > 1:
                duplicate_cells.append({"prompt_id":prompt_id,"kind":kind,"split":split,
                                        "matrix_column":column,"generation_lines_1based":line_numbers})
    coverage = {"created_utc":datetime.now(timezone.utc).isoformat(),
        "generations_file":str(source), "source_snapshot":identity,
        "rows_indexed":len(rows),"arm_groups":len(groups),"prompt_groups":len(prompt_groups),
        "prompts_with_multiple_arm_columns":paired_prompt_count,
        "duplicate_prompt_arm_cells":duplicate_cells,"warnings":warnings,
        "row_reference":"Every generation_line_1based is a physical line in the exact hashed generations.jsonl snapshot.",
        "pairing_identity":["prompt_id","kind","split"],
        "arm_identity":list(ARM_FIELDS),
        "missing_arm_cells":"A blank matrix cell means no saved row was observed for that prompt and arm.",
        "censoring":"I report saved censored and final_ended flags separately; absent flags remain unknown.",
        "provisional_fields":"I copy existing canned-refusal/provisional-success flags as unreviewed proxies in the long index. Their labels are not final judgments.",
        "review_status":{"harmfulness_judging":"unfinished","reasoning_recognition_review":"unfinished"},
        "scope":"I index saved metadata and row locations. Raw responses and emitted reasoning remain in the source file for later review.",
        "arm_columns":{r["matrix_column"]:{f:r[f] for f in ARM_FIELDS} for r in arm_rows}}
    out.mkdir(parents=True,exist_ok=True)
    fields = list(ARM_FIELDS) + ["matrix_column","rows","unique_prompts","final_started","final_ended",
        "final_ended_unknown","censored","censoring_unknown","raw_output_present","cot_present","final_text_present","generation_tokens"]
    write_csv(out/"arm-counts.csv",fields,arm_rows)
    write_csv(out/"prompt-arm-index.csv",INDEX_FIELDS,rows)
    write_csv(out/"prompt-arm-matrix.csv",["prompt_id","kind","split","available_arm_columns",*arm_columns],matrix)
    (out/"coverage.json").write_text(json.dumps(coverage,indent=2,allow_nan=False)+"\n")
    lines = [f"I indexed {len(rows)} saved trajectories across {len(groups)} arm groups and {len(prompt_groups)} prompt groups.",
        "", "Harmfulness judging and review of whether the emitted reasoning recognizes an injection remain unfinished.",
        "", f"The source is [generations.jsonl](<{source}>), SHA256 `{identity['sha256']}`.",
        "", f"- [Arm counts](<{out / 'arm-counts.csv'}>) reports saved completion, censoring, and emitted-reasoning availability.",
        f"- [Prompt and arm index](<{out / 'prompt-arm-index.csv'}>) gives one row per saved trajectory and its source line number.",
        f"- [Pairing matrix](<{out / 'prompt-arm-matrix.csv'}>) places each prompt's available arms side by side. Cell values are source line numbers; blank cells indicate missing rows.",
        f"- [Coverage and column definitions](<{out / 'coverage.json'}>) records the snapshot identity, warnings, duplicate cells, and each matrix column's arm settings.",
        "", "I preserve saved censoring flags separately from final-channel completion. The long index copies any saved provisional labels as unreviewed proxies.",
        "", f"{paired_prompt_count} prompt groups have rows in more than one arm column; {len(duplicate_cells)} prompt-arm cells contain duplicate rows."]
    if warnings:
        lines.extend(["",f"The snapshot has {len(warnings)} parsing warnings, recorded in coverage.json."])
    if rows:
        first=rows[0]["generation_line_1based"]
        lines.extend(["",f"The first indexed trajectory is [source line {first}](<{source}:{first}>)."])
    (out/"README.md").write_text("\n".join(lines)+"\n")
    return coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--allow-partial-tail",action="store_true",
                        help="Record and exclude an incomplete final JSONL line in an in-progress synced snapshot")
    args=parser.parse_args()
    result=build_index(args.generations,args.out,args.allow_partial_tail)
    print(json.dumps({k:result[k] for k in ("rows_indexed","arm_groups","prompt_groups","prompts_with_multiple_arm_columns")}))


if __name__=="__main__":
    main()
