"""
check_references.py
===================
Checks every DOI-bearing reference of the manuscript against Crossref and
flags title or author mismatches (Reviewer #4 point 7 found that reference 24's
DOI belongs to a different paper). Run locally; needs internet access.

    python check_references.py --mailto you@example.org

Output: reference_audit.csv with status OK / TITLE_MISMATCH / AUTHOR_MISMATCH /
DOI_NOT_FOUND / NO_DOI (+ best Crossref bibliographic match for NO_DOI items).
Author surnames are compared after accent stripping; a listed surname absent
from the Crossref record is reported (this catches misspellings).

Known problems found by manual checks during the R3 review (verify, then fix):
  [5]  Issembert: found as a PhilArchive manuscript, not a Philos Technol article
  [8]  "Venkataraman S" should be Venkatasubramanian S (also in [10])
  [11] co-authors should be Bobes-Bascaran J, Fernandez-Leal A (not Mendez,
       Bajo, Corchado); check the issue number
  [12] Harris 2024 BigComp: published title appears to differ
  [14] Kim, Lee, Kim (AIES 2024): not found
  [18] Babich: the located paper is Harvard Review of Philosophy 31:75-97 (2024)
  [24] no paper with these authors/title located; DOI is Henriksen et al.
  [25] the FAO claim (20-40 % less per-hectare allocation) needs a page/passage
"""
import argparse
import csv
import difflib
import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

REFS = [
    (1, ['Mehrabi', 'Morstatter', 'Saxena', 'Lerman', 'Galstyan'], 'A survey on bias and fairness in machine learning', '10.1145/3457607'),
    (2, ['Holstein', 'Wortman Vaughan', 'Daume', 'Wallach'], 'Improving fairness in machine learning systems: What do industry practitioners need?', '10.1145/3290605.3300830'),
    (3, ['Anders'], 'Die Antiquiertheit des Menschen', None),
    (4, ['Anders'], 'Die Antiquiertheit des Menschen II', None),
    (5, ['Issembert'], "The unbearable lightness of a thinking machine: Gunther Anders' Promethean shame in the age of artificial intelligence", '10.1007/s13347-025-00789-5'),
    (6, ['Barocas', 'Hardt', 'Narayanan'], 'Fairness and machine learning: Limitations and opportunities', None),
    (7, ['Kamiran', 'Calders'], 'Data preprocessing techniques for classification without discrimination', '10.1007/s10115-011-0463-8'),
    (8, ['Feldman', 'Friedler', 'Moeller', 'Scheidegger', 'Venkataraman'], 'Certifying and removing disparate impact', '10.1145/2783258.2783311'),
    (9, ['Hardt', 'Price', 'Srebro'], 'Equality of opportunity in supervised learning', None),
    (10, ['Selbst', 'Boyd', 'Friedler', 'Venkataraman', 'Vertesi'], 'Fairness and abstraction in sociotechnical systems', '10.1145/3287560.3287598'),
    (11, ['Mosqueira-Rey', 'Hernandez-Pereira', 'Alonso-Rios', 'Mendez', 'Bajo', 'Corchado'], 'Human-in-the-loop machine learning: A state of the art', '10.1007/s10462-022-10246-w'),
    (12, ['Harris'], 'A new standard for ethical hiring: Combining human judgment with AI fairness tools', '10.1109/BigComp60711.2024.00045'),
    (13, ['Lundberg', 'Lee'], 'A unified approach to interpreting model predictions', None),
    (14, ['Kim', 'Lee', 'Kim'], 'Human-in-the-loop fairness auditing: A case study in hiring', '10.1145/3664647.3680621'),
    (15, ['Copus', 'Spackman', 'Laqueur'], 'Error in the loop: How human mistakes can improve algorithmic learning', '10.5070/L65264123'),
    (16, ['Ghai', 'Mueller'], 'D-BIAS: A causality-based human-in-the-loop system for tackling algorithmic bias', '10.1109/TVCG.2022.3209485'),
    (17, ['Delgado', 'Barocas', 'Levy'], 'Participatory AI design: Lessons from community engagement', '10.1145/3600211.3604689'),
    (18, ['Babich'], "Gunther Anders' Promethean shame and the temporality of technology", '10.1353/sor.2020.0052'),
    (19, ['Lame', 'Dixon-Woods'], 'Using clinical simulation to study how to improve quality and safety in healthcare', '10.1136/bmjstel-2018-000370'),
    (20, ['Fabris', 'Messina', 'Silvello', 'Pasi'], 'Algorithmic fairness datasets: The story so far', '10.1007/s10618-022-00854-9'),
    (21, ['Gemalmaz'], 'Fairness in the machine learning pipeline: A human-in-the-loop perspective', None),
    (22, ['Ribeiro', 'Singh', 'Guestrin'], '"Why should I trust you?" Explaining the predictions of any classifier', '10.1145/2939672.2939778'),
    (23, ['Wachter', 'Mittelstadt', 'Russell'], 'Counterfactual explanations without opening the black box: Automated decisions and the GDPR', '10.2139/ssrn.3063289'),
    (24, ['Zhang', 'Khaliligarekan', 'Tekin', 'Gummadi', 'Weller'], 'Human-in-the-loop fairness: Integrating human feedback for fair decision-making', '10.1145/3461702.3462564'),
    (25, [], 'The State of Food and Agriculture 2022', '10.4060/cb9479en'),
]


def norm(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode()
    return ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split()


def similarity(a, b):
    return difflib.SequenceMatcher(None, ' '.join(norm(a)), ' '.join(norm(b))).ratio()


def get(url, mailto):
    req = urllib.request.Request(url, headers={
        'User-Agent': f'reference-audit/1.0 (mailto:{mailto})'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())['message']


def check(num, authors, title, doi, mailto):
    row = {'ref': num, 'listed_title': title, 'doi': doi or ''}
    try:
        if doi:
            m = get('https://api.crossref.org/works/' + urllib.parse.quote(doi), mailto)
        else:
            q = urllib.parse.quote(title + ' ' + ' '.join(authors))
            items = get('https://api.crossref.org/works?rows=1&query.bibliographic=' + q, mailto)['items']
            if not items:
                return {**row, 'status': 'NO_DOI (no Crossref match)'}
            m = items[0]
    except urllib.error.HTTPError as e:
        return {**row, 'status': 'DOI_NOT_FOUND' if e.code == 404 else f'HTTP {e.code}'}
    except Exception as e:  # network or parse problem
        return {**row, 'status': f'ERROR: {e}'}
    ct = (m.get('title') or [''])[0]
    fam = [' '.join(norm(a.get('family', a.get('name', '')))) for a in m.get('author', [])]
    missing = [a for a in authors if not any(' '.join(norm(a)) in f or f in ' '.join(norm(a)) for f in fam if f)]
    sim = similarity(title, ct)
    status = 'OK'
    if sim < 0.80:
        status = 'TITLE_MISMATCH'
    elif missing and fam:
        status = 'AUTHOR_MISMATCH'
    if not doi:
        status = f'NO_DOI (best match {m.get("DOI", "")}, similarity {sim:.2f})'
    return {**row, 'status': status, 'crossref_title': ct,
            'crossref_authors': '; '.join(fam), 'title_similarity': round(sim, 3),
            'listed_surnames_not_in_crossref': '; '.join(missing),
            'crossref_container': (m.get('container-title') or [''])[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mailto', default='anonymous@example.org')
    ap.add_argument('--out', default='reference_audit.csv')
    args = ap.parse_args()
    rows = []
    for num, authors, title, doi in REFS:
        r = check(num, authors, title, doi, args.mailto)
        rows.append(r)
        print(f"[{num:2d}] {r['status']}")
        time.sleep(1.0)            # be polite to the public API
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'ref', k))
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwritten {args.out}; fix every row whose status is not OK before resubmission')


if __name__ == '__main__':
    main()
