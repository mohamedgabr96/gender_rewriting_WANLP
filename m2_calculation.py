from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
import levenshtein
import os
from utils import smart_open, paragraphs

def normalize(lines):
    """
    Args:
        - lines (list): list of sentences
    Retturns:
        - norm_lines (list): list of normalized lines
    """
    norm_lines = []
    for line in lines:
        norm_line = line.strip()
        norm_line = normalize_alef_ar(norm_line)
        norm_line = normalize_alef_maksura_ar(norm_line)
        norm_line = normalize_teh_marbuta_ar(norm_line)
        norm_lines.append(norm_line)
    return norm_lines


def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, 'r')
    puffer = fgold.read()
    fgold.close()
    puffer = puffer # .decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in list(annotations.keys()):
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def calculate_m2(generated_sents, m2_edits_path, split='dev', specific_direction=None):

    indexes_dict = {"FF": [0, 6647], "FM":[6647, 6647*2] , "MF": [6647*2, 6647*3], "MM": [6647*3, 6647*4]}

    system_sentences = normalize(generated_sents)
    system_sentences = [x.strip() for x in system_sentences]

    m2_edits_final_path = os.path.join(m2_edits_path, split + ".m2")

    source_sentences, gold_edits = load_annotation(m2_edits_final_path)

    if specific_direction is not None:
        index_start = indexes_dict[specific_direction][0]
        index_end = indexes_dict[specific_direction][1]
        source_sentences = source_sentences[index_start:index_end]
        gold_edits = gold_edits[index_start:index_end]


    max_unchanged_words=2
    beta = 0.5
    ignore_whitespace_casing= False
    verbose = False
    very_verbose = False

    p, r, f1 = levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose)

    return p, r, f1
