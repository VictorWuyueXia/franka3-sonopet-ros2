from fr3_sonopet_microphone.audio_format import AudioFormat, describe_audio_format


def test_describe_audio_format_contains_core_fields():
    assert describe_audio_format(AudioFormat()) == "48000Hz/1ch/S16_LE"

