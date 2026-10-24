from meeting_asr.text import disfluency_insensitive, formatting_profile, normalize


def test_disfluency_rule_fillers_and_repeats():
    assert disfluency_insensitive("the the the interaction") == "the interaction"
    assert disfluency_insensitive("i think i think it is er fine") == "i think it is fine"
    assert disfluency_insensitive("a b a b a b c") == "a b c"
    assert disfluency_insensitive("erm") == ""
    assert disfluency_insensitive("yes no yes") == "yes no yes"


def test_reference_and_hypothesis_styles_meet():
    # AMI references are uppercase without punctuation; Whisper output is cased and punctuated.
    assert normalize("DID WE WANNA MEET BEFORE WE TALK TO STEVE") == normalize(
        "Did we wanna meet before we talk to Steve?"
    )


def test_trailing_ellipsis_leaves_no_token():
    assert normalize("Um...") == ""
    assert normalize("So...") == "so"
    assert normalize("It costs 3.50.") == "it costs 3.50"


def test_fillers_removed_and_spelling_unified():
    assert normalize("UM so UH the colour") == normalize("so the color")


def test_formatting_profile():
    p = formatting_profile(["Hello, world.", "NO PUNCT HERE", ""])
    assert p["frac_with_punct"] == 0.5
    assert p["frac_with_lowercase"] == 0.5
