# TOY LAB (MATS 12.0 application experiment). Fixtures are public Wikipedia snapshots with the paper's frozen
# injection templates; the "secret" is a random hex string that unlocks nothing. See ../../TOY-LAB-NOTICE.md.
"""Build the page set: N frozen Wikipedia pages x {standard, forgery} using the frozen page builder.

  python pages.py --out data/pages-24 --pages 24 --seed 2026            # fetch on the pod (network needed)
  python pages.py --out data/pages-5 --from-local ../agent-hijacking/data/pilot-20260911   # reuse the pilot's 5 pages

Output: <out>/manifest.json with cases (id, page_id, variant, template_type, payload, fixture_path, user_prompt),
fixtures/, raw/, and source/ (the authors' injections.yaml copied verbatim). Template types cycle through the
paper's five templates, so with 24 pages each template appears on 4 or 5 pages and page/template stop being
confounded.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-candidates", type=int, default=800)
    ap.add_argument("--source", type=Path, default=Path("/workspace/prompt-injection-as-role-confusion"))
    ap.add_argument("--from-local", type=Path, default=None, help="copy an existing pilot manifest instead of fetching")
    args = ap.parse_args()
    if (args.out / "manifest.json").exists():
        raise SystemExit("manifest exists; use a new --out")
    args.out.mkdir(parents=True)
    if args.from_local:
        src = args.from_local
        for d in ("fixtures", "raw", "source"):
            shutil.copytree(src / d, args.out / d)
        m = json.loads((src / "manifest.json").read_text())
        m["copied_from"] = str(src)
        (args.out / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"pages": len(m["pages"]), "cases": len(m["cases"]), "out": str(args.out)}))
        return
    import sys
    sys.path.insert(0, str(HERE))
    import harness_prepare as hp
    pages = hp.freeze_pages(args.out, args.pages, args.seed, args.max_candidates)
    manifest = hp.build_manifest(args.out, pages, args.source, args.seed)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pages": len(pages), "cases": len(manifest["cases"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
