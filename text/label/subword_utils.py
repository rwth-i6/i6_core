import xml.etree.ElementTree as ET


def get_lm_tokens_from_lexicon(lexicon):
    """
    This function extracts LM tokens (vocabularies) from a given lexicon and returns the tokens as a
    list of strings.
    :param Lexicon lexicon: represents a bliss lexicon
    """
    lm_tokens = set()
    for l in lexicon.lemmata:
        for orth in l.orth:
            lm_tokens.add(orth)
        for token in l.synt or []:  # l.synt can be None
            lm_tokens.add(token)
        for eval in l.eval:
            for t in eval:
                lm_tokens.add(t)

    return list(lm_tokens)


def word_to_subword_in_lexicon(lexicon, lm_tokens, subword_tokens):
    """
    This function converts the word-level tokens in a lexicon to sub-word-level tokens given the
    appropriate lexicon, list of word-level and sub-word-level tokens. It returns an XML ElementTree
    containing all the information of the lexicon.
    :param Lexicon lexicon: represents a bliss lexicon
    :param List[str] lm_tokens: A list of word-level tokens (vocabularies) obtained from the lexicon
    :param List[str] subword_tokens: A list of subword tokens obtained from the lm_tokens using
    appropriate techniques (bpe, spm, etc.). The two lists should be of equal length.
    """
    assert len(lm_tokens) == len(
        subword_tokens
    ), f"The length of lm_tokens {len(lm_tokens)} does not match the length of subword_tokens {len(subword_tokens)}."

    w2s = dict(zip(lm_tokens, subword_tokens))

    for l in lexicon.lemmata:
        if l.special is None and len(l.orth) > 0:
            if not l.synt and len(l.eval) == 0:
                o = l.orth[0]
                l.synt = w2s[o]
                l.eval.append([o])
            if l.synt:
                l.synt = sum([w2s[token] for token in l.synt], [])
            if len(l.eval) > 0:
                l.eval = [sum([w2s[t] for t in token_sequence], []) for token_sequence in l.eval]

    elem = lexicon.to_xml()
    return ET.ElementTree(elem)
