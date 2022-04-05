"""
This is an old location of bpe jobs kept for backwards compatibility, for new setups using the subword-nmt based BPE,
please use i6_core.label.bpe, for other setups please switch to the sentencepiece implementation
"""
from i6_core.text.label.subword_nmt.apply import ApplyBPEModelToLexiconJob  # noqa
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob  # noqa
