"""Apply manual seed annotations to the exported YAML files."""
import yaml
from pathlib import Path

HEADER = """# Pronoun Recovery — Manual Seed Annotations
#
# Marker format:  [PRO.<label>:<form>]
# Labels: PRO.1sg, PRO.2sg, PRO.3sg, PRO.1pl, PRO.2pl, PRO.3pl, IMP, CONJ
#
# Leave 'annotated' unchanged if no pronouns are dropped.
# Do NOT edit the 'id', 'genre', or 'text' fields.
#

"""

# Each value is a list of (find, replace) applied sequentially.
IT_ANNOTATIONS = {
    "childes:chunk_2619": [
        ("allora la richiudo", "allora [PRO.1sg:io] la richiudo"),
        ("gli fai un po' di spazio", "[PRO.2sg:tu] gli fai un po' di spazio"),
        (", facciamo girare", ", [PRO.1pl:noi] facciamo girare"),
        ("uh , sei riuscito", "uh , [PRO.2sg:tu] sei riuscito"),
        ("adesso vediamo se riesci", "adesso [PRO.1pl:noi] vediamo se [PRO.2sg:tu] riesci"),
    ],
    "clta:chunk_209": [
        ("che suggerisco", "che [PRO.1sg:io] suggerisco"),
        ("prima ti distrai", "prima [PRO.2sg:tu] ti distrai"),
        ("in seguito rileggi", "in seguito [PRO.2sg:tu] rileggi"),
        ("e lo controlli", "e [PRO.2sg:tu] lo controlli"),
        ("controlli ti auguro", "controlli [PRO.1sg:io] ti auguro"),
        ("aiutato e spero", "aiutato e [PRO.1sg:io] spero"),
    ],
    "europarl:chunk_16469": [
        ("Vi ringrazio,", "[PRO.1sg:io] Vi ringrazio,"),
        ("colleghi, devo", "colleghi, [PRO.1sg:io] devo"),
        ("che abbiamo più", "che [PRO.1pl:noi] abbiamo più"),
        ("Parlamento. Stiamo", "Parlamento. [PRO.1pl:noi] Stiamo"),
        ("mancanti. Spero", "mancanti. [PRO.1sg:io] Spero"),
        ("nominale. Procederemo", "nominale. [PRO.1pl:noi] Procederemo"),
        ("quando disporremo", "quando [PRO.1pl:noi] disporremo"),
    ],
    "europarl:chunk_44907": [
        ("animali. Dobbiamo", "animali. [PRO.1pl:noi] Dobbiamo"),
        ("oggi, potremmo", "oggi, [PRO.1pl:noi] potremmo"),
    ],
    "europarl:chunk_53587": [
        ("relatrice, ho ascoltato", "relatrice, [PRO.1sg:io] ho ascoltato"),
    ],
    "corpus_isacco:chunk_65": [
        ("quindi  glielo  tirarono", "quindi  [PRO.3pl:loro] glielo  tirarono"),
        ("poi glielo fecero", "poi [PRO.3pl:loro] glielo fecero"),
        ("non lo  presero", "[PRO.3pl:loro] non lo  presero"),
        ("presero perch era", "presero perch [PRO.3sg:lui] era"),
        ("poi channo fatto", "poi [PRO.3pl:loro] channo fatto"),
        ("rondini e hanno lasciato", "rondini e [PRO.3pl:loro] hanno lasciato"),
    ],
    "leipzig_web:chunk_35509": [
        ("attesa: sembrano", "attesa: [PRO.3pl:loro] sembrano"),
    ],
    "leipzig_web:chunk_90519": [
        ("fa; ma dobbiamo", "fa; ma [PRO.1pl:noi] dobbiamo"),
    ],
    "qcri:chunk_45178": [
        ("che spero di fare", "che [PRO.1sg:io] spero di fare"),
        ("se entrate in", "se [PRO.2pl:voi] entrate in"),
        ("cinese, potreste", "cinese, [PRO.2pl:voi] potreste"),
        ("Stasera voglio", "Stasera [PRO.1sg:io] voglio"),
        ("che se studiamo", "che se [PRO.1pl:noi] studiamo"),
        ("pipistrelli saremo", "pipistrelli [PRO.1pl:noi] saremo"),
    ],
    "qcri:chunk_6723": [
        ("opinioni -- avrebbero", "opinioni -- [PRO.3pl:loro] avrebbero"),
        ("No, hanno buttato", "No, [PRO.3pl:loro] hanno buttato"),
        ("E cosa ne dite", "E [PRO.2pl:voi] cosa ne dite"),
        ("Se scorrete", "Se [PRO.2pl:voi] scorrete"),
        ("Quindi cosa dobbiamo", "Quindi [PRO.1pl:noi] cosa dobbiamo"),
        ("Abbiamo bisogno", "[PRO.1pl:noi] Abbiamo bisogno"),
    ],
    "spgc:chunk_25972": [
        ("ma che si consigliasse", "ma che [PRO.3sg:lui] si consigliasse"),
    ],
}

EN_ANNOTATIONS = {
    "childes:chunk_11557": [
        ("[grunts]. told you", "[grunts]. [PRO.1sg:I] told you"),
    ],
    "open_subtitles:chunk_34687": [
        ("foreigner, came from", "foreigner, [PRO.1sg:I] came from"),
    ],
    "open_subtitles:chunk_38974": [
        ("What? Said 2-Z", "What? [PRO.3sg:she] Said 2-Z"),
    ],
    "open_subtitles:chunk_72768": [
        ('kids." Makes you', 'kids." [PRO.3sg:it] Makes you'),
    ],
}


def apply_annotations(yaml_path, annotations):
    with open(yaml_path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    n_applied = 0
    for entry in entries:
        eid = entry["id"]
        if eid in annotations:
            annotated = entry["annotated"]
            for find, replace in annotations[eid]:
                if find not in annotated:
                    print(f"  WARNING: '{find}' not found in {eid}")
                else:
                    annotated = annotated.replace(find, replace, 1)
                    n_applied += 1
            entry["annotated"] = annotated

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(HEADER)
        yaml.dump(
            entries, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    n_changed = len(annotations)
    n_unchanged = len(entries) - n_changed
    print(f"{yaml_path.name}: {n_applied} markers in {n_changed} entries, {n_unchanged} unchanged")


it_path = Path("data/pronoun_recovery/annotations/it/manual/seed_annotations.yaml")
en_path = Path("data/pronoun_recovery/annotations/en/manual/seed_annotations.yaml")

apply_annotations(it_path, IT_ANNOTATIONS)
apply_annotations(en_path, EN_ANNOTATIONS)
