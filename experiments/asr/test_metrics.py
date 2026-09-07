"""Tests for metrics.py, each with a case that must FAIL if the metric is wrong."""
from metrics import (amplification, build_rare_vocab, multiset_recall, norm_words,
                     score_pair, wer)

def t(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return cond

ok = True
# norm_words erases surface form only.
ok &= t("norm strips case and punctuation",
        norm_words('He said, "Hello-World!"') == ["he", "said", "hello", "world"],
        str(norm_words('He said, "Hello-World!"')))
ok &= t("norm makes whisper and librispeech comparable",
        norm_words('"Give not so earnest a mind."') == norm_words("GIVE NOT SO EARNEST A MIND"))
ok &= t("norm keeps digits", norm_words("in 1947 he left") == ["in", "1947", "he", "left"])

# multiset recall must count repetition, not just presence.
ok &= t("multiset recall counts repeats", multiset_recall(["a","a","b"], ["a","b"]) == 2/3,
        str(multiset_recall(["a","a","b"], ["a","b"])))
ok &= t("set-recall would wrongly say 1.0 here",
        multiset_recall(["a","a","b"], ["a","b"]) != 1.0)
ok &= t("identical -> 1.0", multiset_recall(["a","b"], ["a","b"]) == 1.0)
ok &= t("disjoint -> 0.0", multiset_recall(["a"], ["b"]) == 0.0)
ok &= t("empty ref -> 1.0", multiset_recall([], ["b"]) == 1.0)
ok &= t("insertions do not raise recall", multiset_recall(["a"], ["a","x","y"]) == 1.0)

# rare vocab is reference-only and frequency-based.
rare, counts = build_rare_vocab(["the cat sat", "the dog sat", "the cat ran"], rare_max=1)
ok &= t("frequent words excluded from rare set", "the" not in rare and "sat" not in rare and "cat" not in rare)
ok &= t("singleton words are rare", {"dog", "ran"} <= rare, str(sorted(rare)))

# score_pair: a hypothesis that drops a rare word is penalised on rare_recall
# much harder than on word_recall.
rare2, _ = build_rare_vocab(["the man saw the ship at dover the man ran"], rare_max=1)
s_full = score_pair("the man saw the ship at dover", "the man saw the ship at dover", rare2)
s_drop = score_pair("the man saw the ship at dover", "the man saw the ship at the", rare2)
ok &= t("perfect copy scores 1.0 on both", s_full["word_recall"] == 1.0 and s_full["rare_recall"] == 1.0)
# 'dover' is one of FOUR rare words in that reference (saw/ship/at/dover), so
# dropping it costs 1/4 of rare_recall but only 1/7 of word_recall. The point of
# the rare band is exactly this: it is more sensitive to a lost content word.
ok &= t("dropping a rare word costs more on rare_recall than on word_recall",
        abs(s_drop["rare_recall"] - 0.75) < 1e-9
        and abs(s_drop["word_recall"] - 6/7) < 1e-9
        and s_drop["rare_recall"] < s_drop["word_recall"],
        f"rare={s_drop['rare_recall']:.3f} word={s_drop['word_recall']:.3f}")
# and dropping a FREQUENT word must not move rare_recall at all
s_freq = score_pair("the man saw the ship at dover", "man saw the ship at dover", rare2)
ok &= t("dropping a frequent word leaves rare_recall untouched",
        s_freq["rare_recall"] == 1.0 and s_freq["word_recall"] < 1.0,
        f"rare={s_freq['rare_recall']:.3f} word={s_freq['word_recall']:.3f}")

# amplification semantics.
ok &= t("transparent compression -> A == 1", amplification(0.20, 0.20) == 1.0)
ok &= t("amplifying compression -> A > 1", amplification(0.20, 0.40) == 2.0)
ok &= t("absorbing compression -> A < 1", amplification(0.20, 0.10) == 0.5)
ok &= t("near-clean L0 refuses to divide", amplification(0.001, 0.5) is None)

# WER against a hand-checked case.
w = wer("the quick brown fox", "the quick brown fox")
ok &= t("wer 0 on identical", w["wer"] == 0.0)
w = wer("A B C D", "A X C")           # 1 substitution + 1 deletion out of 4
ok &= t("wer counts S and D", abs(w["wer"] - 0.5) < 1e-9 and w["sub"] == 1 and w["dele"] == 1,
        str(w))
w = wer("A B", "A B C")               # 1 insertion
ok &= t("wer counts insertions", abs(w["wer"] - 0.5) < 1e-9 and w["ins"] == 1, str(w))

print(f"\n{'ALL METRIC TESTS PASS' if ok else 'METRIC TESTS FAILED'}")
raise SystemExit(0 if ok else 1)
