"""
Convert raw Synthea FHIR R4 bundles (one JSON per patient, multi-site) into
CSV tables that match Synthea's own CSV export schema, so eda_los.py and later
steps run unchanged.

Reads ONLY the raw staging copy (read-only). Never touches MinIO/bronze/silver.

Usage:
  python fhir_to_tables.py \
      --raw ~/Desktop/PoisonWatch/data/synthea_raw \
      --out data/synthea_tables/csv
Expects: <raw>/<site_id>/output/fhir/*.json
"""
import argparse, csv, json, sys, time
from pathlib import Path

CLASS_MAP = {"IMP": "inpatient", "AMB": "ambulatory", "EMER": "emergency",
             "ACUTE": "inpatient", "NONAC": "inpatient", "HH": "home", "VR": "virtual"}

def ref_id(ref):
    """'urn:uuid:abc' or 'Patient/abc' -> 'abc'"""
    if not ref:
        return ""
    return ref.rsplit(":", 1)[-1].rsplit("/", 1)[-1]

def code_display(cc):
    if not cc:
        return "", ""
    c = (cc.get("coding") or [{}])[0]
    return c.get("code", ""), cc.get("text") or c.get("display", "")

def category(obs):
    cats = obs.get("category") or []
    if cats and cats[0].get("coding"):
        return cats[0]["coding"][0].get("code", "")
    return ""

def obs_rows(o):
    """Yield (CODE, DESCRIPTION, VALUE, UNITS, TYPE). Splits component panels (e.g. BP)."""
    parts = o.get("component") if o.get("component") else [o]
    for p in parts:
        code, desc = code_display(p.get("code"))
        if "valueQuantity" in p:
            vq = p["valueQuantity"]
            yield code, desc, vq.get("value", ""), vq.get("unit", ""), "numeric"
        elif "valueCodeableConcept" in p:
            _, vd = code_display(p["valueCodeableConcept"])
            yield code, desc, vd, "", "text"
        elif "valueString" in p:
            yield code, desc, p["valueString"], "", "text"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/synthea_tables/csv")
    args = ap.parse_args()
    raw = Path(args.raw).expanduser()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in raw.glob("*/output/fhir/*.json")
                   if not f.name.startswith(("hospitalInformation", "practitionerInformation")))
    if not files:
        sys.exit(f"No bundles found under {raw}/<site_id>/output/fhir/")
    print(f"Found {len(files):,} patient bundles")

    fp = open(out / "patients.csv", "w", newline="")
    fe = open(out / "encounters.csv", "w", newline="")
    fo = open(out / "observations.csv", "w", newline="")
    wp, we, wo = csv.writer(fp), csv.writer(fe), csv.writer(fo)
    wp.writerow(["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "SITE_ID"])
    we.writerow(["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE", "DESCRIPTION", "SITE_ID"])
    wo.writerow(["DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"])

    n = {"pat": 0, "enc": 0, "obs": 0, "poisoned_flag": 0, "bad_files": 0}
    unknown_classes = {}
    t0 = time.time()
    for i, f in enumerate(files, 1):
        site = f.parts[-4]  # <site_id>/output/fhir/<file>
        try:
            bundle = json.loads(f.read_bytes())
        except Exception as e:
            n["bad_files"] += 1
            print(f"  skip {f.name}: {e}")
            continue
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            if (r.get("meta") or {}).get("is_poisoned") is True:
                n["poisoned_flag"] += 1
            rt = r.get("resourceType")
            if rt == "Patient":
                wp.writerow([r.get("id"), r.get("birthDate", ""),
                             r.get("deceasedDateTime", ""), r.get("gender", ""), site])
                n["pat"] += 1
            elif rt == "Encounter":
                cls = (r.get("class") or {}).get("code", "")
                if cls not in CLASS_MAP:
                    unknown_classes[cls] = unknown_classes.get(cls, 0) + 1
                code, desc = code_display((r.get("type") or [{}])[0])
                per = r.get("period") or {}
                we.writerow([r.get("id"), per.get("start", ""), per.get("end", ""),
                             ref_id((r.get("subject") or {}).get("reference")),
                             CLASS_MAP.get(cls, cls.lower()), code, desc, site])
                n["enc"] += 1
            elif rt == "Observation":
                pid = ref_id((r.get("subject") or {}).get("reference"))
                eid = ref_id((r.get("encounter") or {}).get("reference"))
                date = r.get("effectiveDateTime", "")
                cat = category(r)
                for code, desc, val, unit, typ in obs_rows(r):
                    wo.writerow([date, pid, eid, cat, code, desc, val, unit, typ])
                    n["obs"] += 1
        if i % 1000 == 0:
            print(f"  {i:,}/{len(files):,} bundles  ({time.time()-t0:.0f}s)", flush=True)

    for fh in (fp, fe, fo):
        fh.close()
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Patients {n['pat']:,} | Encounters {n['enc']:,} | Observation rows {n['obs']:,}")
    if unknown_classes:
        print(f"Unmapped encounter class codes (kept lowercase): {unknown_classes}")
    if n["bad_files"]:
        print(f"WARNING: {n['bad_files']} unreadable files skipped")
    if n["poisoned_flag"]:
        print(f"WARNING: {n['poisoned_flag']} resources have meta.is_poisoned=true. "
              "This is NOT a clean copy. Stop and point --raw at untouched Synthea output.")
    else:
        print("Clean check passed: no resources flagged is_poisoned=true")

if __name__ == "__main__":
    main()
