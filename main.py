import csv
import logging
import unicodedata
import re
from difflib import SequenceMatcher
from collections import defaultdict

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load CRSP data
crsp_rows = []
logger.info('Loading CRSP data...')
with open('Data/CRSPnames.csv', newline='', encoding='utf-8') as crsp_file:
    reader = csv.DictReader(crsp_file)
    for row in tqdm(reader, desc='CRSP rows', unit='row'):
        crsp_rows.append(row)
logger.info('Loaded %d CRSP rows.', len(crsp_rows))

# Helper functions for text processing
def normalize_text(text):
    if text is None:
        return ''
    text = str(text)
    if not text:
        return ''
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.upper()
    text = text.replace('&', ' AND ')
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = text.split()
    stopwords = {
        'INC', 'CORP', 'CORPORATION', 'LTD', 'LIMITED', 'PLC', 'SA', 'NV', 'AG', 'CO', 'COMPANY',
        'HOLDINGS', 'HOLDING', 'GROUP', 'THE', 'INCORPORATED', 'LLC', 'LP', 'SPA', 'AB', 'SE',
        'OYJ', 'ASA', 'BV', 'KK', 'PUBLIC', 'CORP', 'COOP', 'AS', 'NV', 'PLC', 'S', 'A'
    }
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(cleaned_tokens).strip()

def extract_company_from_headline(headline, fallback):
    text = '' if headline is None else str(headline)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.upper()
    cutoff_keywords = ['EARNINGS', 'RESULTS', 'CONFERENCE', 'CALL', 'WEBCAST', 'TRADING', 'MEETING', 'PRESENTATION', 'UPDATE']
    cut_position = len(text)
    for keyword in cutoff_keywords:
        idx = text.find(keyword)
        if idx != -1:
            cut_position = min(cut_position, idx)
    text = text[:cut_position]
    tokens = re.split(r'[^A-Z0-9&]+', text)
    prefix_stopwords = {
        'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'FISCAL', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'HALF',
        'FULL', 'YEAR', 'YEARS', 'ANNUAL', 'INTERIM', 'H1', 'H2', 'H3', 'H4', 'NINE', 'MONTH',
        'MONTHS', 'SEVEN', 'EIGHT', 'TEN', 'ELEVEN', 'TWELVE', 'RESULTS', 'EARNINGS',
        'CONFERENCE', 'CALL', 'TRANSCRIPT', 'WEBCAST', 'UPDATE', 'PRESS', 'RELEASE', 'INVESTOR',
        'DAY', 'MEETING', 'FIRSTQUARTER', 'SECONDQUARTER', 'THIRDQUARTER', 'FOURTHQUARTER'
    }
    cleaned_tokens = []
    prefix_skipped = False
    for token in tokens:
        if not token:
            continue
        if not prefix_skipped:
            if token in prefix_stopwords or re.fullmatch(r'\d{1,4}', token):
                continue
            prefix_skipped = True
        cleaned_tokens.append(token)
    if not cleaned_tokens:
        fallback_text = '' if fallback is None else str(fallback)
        fallback_text = unicodedata.normalize('NFKD', fallback_text)
        fallback_text = fallback_text.encode('ascii', 'ignore').decode('ascii')
        return fallback_text
    return ' '.join(cleaned_tokens)

# Build lookup structures for CRSP names
normalized_to_comnam = defaultdict(list)
first_char_index = defaultdict(list)
token_index = defaultdict(list)
all_candidates = []
norm_tokens_map = {}
logger.info('Building lookup structures for CRSP names...')
for row in tqdm(crsp_rows, desc='Indexing CRSP names', unit='row'):
    comnam = row.get('COMNAM', '')
    norm = normalize_text(comnam)
    if not norm:
        continue
    normalized_to_comnam[norm].append(comnam)
    first_char = norm[0]
    first_char_index[first_char].append((norm, comnam))
    tokens = norm.split()
    token_set = tuple(tokens)
    norm_tokens_map[norm] = set(tokens)
    for token in tokens:
        token_index[token].append((norm, comnam))
    all_candidates.append((norm, comnam))

# Load SE data and attempt matching
processed_rows = []
unmatched_rows = []
unmatched_signatures = set()
match_count = 0
unmatched_count = 0
total_rows = 0
logger.info('Processing SE mappings for matching...')
with open('Data/SEmappingsDAFA.csv', newline='', encoding='utf-8') as se_file:
    reader = csv.DictReader(se_file)
    base_fieldnames = reader.fieldnames if reader.fieldnames is not None else []
    fieldnames = base_fieldnames + ['ExtractedCompanyName', 'NormalizedCompanyName', 'MergeComnam']
    signature_fields = base_fieldnames + ['NormalizedCompanyName']
    for row in tqdm(reader, desc='Matching SE rows', unit='row'):
        total_rows += 1
        extracted = extract_company_from_headline(row.get('SEHeadline', ''), row.get('SECompanyName', ''))
        normalized = normalize_text(extracted)
        matched_comnam = ''
        if normalized in normalized_to_comnam:
            matched_comnam = normalized_to_comnam[normalized][0]
        else:
            candidate_set = set()
            tokens = normalized.split()
            token_set = set(tokens)
            for token in tokens:
                for candidate in token_index.get(token, []):
                    candidate_set.add(candidate)
            if not candidate_set and normalized:
                first_char = normalized[0]
                for candidate in first_char_index.get(first_char, []):
                    candidate_set.add(candidate)
            if not candidate_set:
                candidate_set = set(all_candidates)
            scored_candidates = []
            for candidate_norm, candidate_comnam in candidate_set:
                candidate_tokens = norm_tokens_map.get(candidate_norm, set())
                if not tokens and not candidate_tokens:
                    token_score = 0.0
                else:
                    intersection = len(token_set & candidate_tokens)
                    denominator = max(len(token_set), len(candidate_tokens), 1)
                    token_score = intersection / denominator
                scored_candidates.append((token_score, candidate_norm, candidate_comnam))
            scored_candidates.sort(reverse=True)
            best_match = None
            best_score = 0.0
            for token_score, candidate_norm, candidate_comnam in scored_candidates[:20]:
                if normalized and (normalized in candidate_norm or candidate_norm in normalized):
                    score = 1.0
                else:
                    if token_score < 0.3:
                        continue
                    score = SequenceMatcher(None, normalized, candidate_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_match = candidate_comnam
            if best_match is None and scored_candidates:
                best_match = scored_candidates[0][2]
                best_score = scored_candidates[0][0]
            if best_match is not None and best_score >= 0.8:
                matched_comnam = best_match
        output_row = dict(row)
        output_row['ExtractedCompanyName'] = extracted
        output_row['NormalizedCompanyName'] = normalized
        output_row['MergeComnam'] = matched_comnam
        if matched_comnam:
            match_count += 1
        else:
            unmatched_count += 1
            signature = tuple(output_row.get(field, '') for field in signature_fields)
            if signature not in unmatched_signatures:
                unmatched_signatures.add(signature)
                unmatched_rows.append(output_row)
        processed_rows.append(output_row)

# Write results to file
output_path = 'Data/SEmappingsDAFA_merged.csv'
logger.info('Writing merged results to %s', output_path)
with open(output_path, 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(processed_rows)

# Write unmatched unique rows to a separate file
unmatched_output_path = 'Data/SEmappingsDAFA_unmatched_unique.csv'
logger.info('Writing %d unique unmatched rows (from %d unmatched) to %s',
            len(unmatched_rows), unmatched_count, unmatched_output_path)
with open(unmatched_output_path, 'w', newline='', encoding='utf-8') as unmatched_file:
    writer = csv.DictWriter(unmatched_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(unmatched_rows)

# Print merge success rate
if total_rows:
    success_rate = match_count / total_rows * 100
else:
    success_rate = 0.0
logger.info('Matched %d out of %d records (%.2f%% success rate).', match_count, total_rows, success_rate)
